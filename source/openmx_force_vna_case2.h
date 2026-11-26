#ifndef OPENMX_FORCE_VNA_CASE2_H
#define OPENMX_FORCE_VNA_CASE2_H
#include "openmx_core_vna_matrix.h"
#include <string.h>

/* Force4B case 2 needs the projector rows of each centre's neighbours,
   including the reverse periodic images. Use the existing first-neighbour
   halo to exchange the owner's local centre index once. The force loop
   can then exchange only the four projector planes scheduled for a turn. */
typedef struct {
    int local_count, neighbour_capacity;
    const int *local_atom, *force_index, *species, *orbitals, *neighbour_count;
    int *const *neighbours;
    const int *sent_atoms, *received_atoms, *receive_start;
    int *const *sent_local;
    int *const *centre, *const *source_atom, *const *source_neighbour, *const *target_neighbour;
} OpenMXVnaCase2Topology;

typedef struct {
    OpenMXVnaCase2Topology topology;
    OpenMXProjectorTransfer *transfer;
    MPI_Datatype datatype;
    int rank, peers, tag, projectors;
    size_t storage_count;
    int *turn, *owner, *receive_head, *receive_next, *send_entry;
} OpenMXVnaCase2Exchange;

static inline int OpenMX_VnaCase2Orbitals(const OpenMXVnaCase2Topology *t, int atom)
{
    return t->orbitals[t->species[atom]];
}

static inline void OpenMX_VnaCase2SendRecv(const OpenMXVnaCase2Exchange *x,
    const void *send, int sent, int destination, void *receive, int received, int source,
    MPI_Datatype datatype)
{
    if (!sent && !received) return;
    MPI_Status status;
    if (MPI_Sendrecv(send, sent, datatype, sent ? destination : MPI_PROC_NULL, x->tag,
        receive, received, datatype, received ? source : MPI_PROC_NULL, x->tag,
        x->transfer->communicator, &status) != MPI_SUCCESS)
        OpenMX_TransferFail(x->transfer->communicator, "VNA case-2 exchange failed");
    if (received) {
        int count;
        if (MPI_Get_count(&status, datatype, &count) != MPI_SUCCESS || count != received)
            OpenMX_TransferFail(x->transfer->communicator, "inconsistent VNA case-2 message size");
    }
}

static inline void OpenMX_VnaCase2Begin(OpenMXVnaCase2Exchange *x, OpenMXSolverInvocation *scratch)
{
    const OpenMXVnaCase2Topology *t = &x->topology;
    size_t atoms = (size_t)t->local_count + 1;
    for (int peer = 0; peer < x->peers; ++peer)
        if (peer != x->rank) atoms += (size_t)t->received_atoms[peer];
    x->storage_count = atoms;
    x->turn = OpenMX_Solver_Int(scratch, "Force4B:centre-turn", atoms);
    x->owner = OpenMX_Solver_Int(scratch, "Force4B:centre-owner", atoms);
    x->receive_head = OpenMX_Solver_Int(scratch, "Force4B:receive-head", x->peers);
    x->receive_next = OpenMX_Solver_Int(scratch, "Force4B:receive-next", t->neighbour_capacity);
    x->send_entry = OpenMX_Solver_Int(scratch, "Force4B:send-entry", t->neighbour_capacity);
    for (size_t i = 0; i < atoms; ++i) {
        x->turn[i] = 0;
        x->owner[i] = -1;
    }
    for (int mc = 1; mc <= t->local_count; ++mc) {
        x->turn[mc] = mc;
        x->owner[mc] = x->rank;
    }
    for (int shift = 1; shift < x->peers; ++shift) {
        const int destination = (x->rank + shift) % x->peers;
        const int source = (x->rank - shift + x->peers) % x->peers;
        const int received = t->received_atoms[source];
        const int first = received ? t->receive_start[source] : 0;
        if (received && (first <= t->local_count || (size_t)first + received > atoms))
            OpenMX_TransferFail(x->transfer->communicator, "invalid VNA case-2 halo range");
        OpenMX_VnaCase2SendRecv(x, t->sent_local[destination], t->sent_atoms[destination],
            destination, x->turn + first, received, source, MPI_INT);
        for (int i = 0; i < received; ++i) x->owner[first + i] = source;
    }
    for (int peer = 0; peer < x->peers; ++peer) {
        x->transfer->buffers.sent[peer] = 0;
        /* The send lists are sorted by global centre, as are their
           owners' local atom lists. Validate the resulting turn order. */
        int previous = 0;
        const int *centres = t->centre[peer];
        for (int p = 0; centres[p] != 0;) {
            const int atom = centres[p];
            const int mc = t->force_index[atom];
            if (mc < 1 || (size_t)mc >= atoms || x->owner[mc] != peer || x->turn[mc] <= previous)
                OpenMX_TransferFail(x->transfer->communicator, "invalid VNA case-2 centre schedule");
            previous = x->turn[mc];
            do { ++p; } while (centres[p] == atom);
        }
    }
}

static inline int OpenMX_VnaCase2Count(OpenMXVnaCase2Exchange *x, int total, int atom)
{
    const int count = OpenMX_TransferCount(x->transfer, 4,
        OpenMX_VnaCase2Orbitals(&x->topology, atom), x->projectors);
    if (count > INT_MAX - total)
        OpenMX_TransferFail(x->transfer->communicator, "VNA case-2 message exceeds MPI count");
    return total + count;
}

/* The wire stores rows in the receiving centre's neighbour order. Both
   peers know the orbital counts and which neighbours belong to each rank. */
static inline int OpenMX_VnaCase2SendPlan(OpenMXVnaCase2Exchange *x, int peer, int turn, int *centre)
{
    const OpenMXVnaCase2Topology *t = &x->topology;
    int p = x->transfer->buffers.sent[peer];
    const int atom = t->centre[peer][p];
    *centre = 0;
    if (!atom || x->turn[t->force_index[atom]] != turn) return 0;
    *centre = atom;
    const int neighbours = t->neighbour_count[atom] + 1;
    if (neighbours > t->neighbour_capacity)
        OpenMX_TransferFail(x->transfer->communicator, "invalid VNA case-2 neighbour count");
    for (int k = 0; k < neighbours; ++k) x->send_entry[k] = -1;
    int length = 0;
    for (; t->centre[peer][p] == atom; ++p) {
        const int k = t->target_neighbour[peer][p];
        if (k < 0 || k >= neighbours || x->send_entry[k] != -1)
            OpenMX_TransferFail(x->transfer->communicator, "invalid VNA case-2 reverse neighbour");
        x->send_entry[k] = p;
        length = OpenMX_VnaCase2Count(x, length, t->neighbours[atom][k]);
    }
    x->transfer->buffers.sent[peer] = p;
    return length;
}

static inline void OpenMX_VnaCase2Stage(OpenMXVnaCase2Exchange *x, int turn,
    const OpenMXFloatBlockMatrix *projectors)
{
    const OpenMXVnaCase2Topology *t = &x->topology;
    const int centre = turn <= t->local_count ? t->local_atom[turn] : 0;
    const int slot = t->local_count + 1;
    int *receive_counts = x->transfer->buffers.received;
    for (int peer = 0; peer < x->peers; ++peer) {
        x->receive_head[peer] = -1;
        receive_counts[peer] = 0;
    }
    if (centre) {
        const int neighbours = t->neighbour_count[centre] + 1;
        if (neighbours > t->neighbour_capacity)
            OpenMX_TransferFail(x->transfer->communicator, "invalid VNA case-2 neighbour count");
        for (int k = neighbours - 1; k >= 0; --k) {
            const int atom = t->neighbours[centre][k], mc = t->force_index[atom];
            const int peer = mc < 1 || (size_t)mc >= x->storage_count ? -1 : x->owner[mc];
            if (peer < 0)
                OpenMX_TransferFail(x->transfer->communicator, "missing VNA case-2 neighbour owner");
            x->receive_next[k] = x->receive_head[peer];
            x->receive_head[peer] = k;
            receive_counts[peer] = OpenMX_VnaCase2Count(x, receive_counts[peer], atom);
        }
    }
    for (int shift = 0; shift < x->peers; ++shift) {
        const int destination = (x->rank + shift) % x->peers;
        const int source = (x->rank - shift + x->peers) % x->peers;
        int target = 0;
        const int sent = OpenMX_VnaCase2SendPlan(x, destination, turn, &target);
        const int received = receive_counts[source];
        if (!sent && !received) continue;
        if (!shift && (target != centre || sent != received))
            OpenMX_TransferFail(x->transfer->communicator, "inconsistent local VNA case-2 neighbours");
        float *send = shift && sent
            ? OpenMX_TransferBuffer(x->transfer, OPENMX_EXCHANGE_SEND, sent) : NULL;
        float *receive = shift && received
            ? OpenMX_TransferBuffer(x->transfer, OPENMX_EXCHANGE_RECEIVE, received) : NULL;
        float *wire = send;
        if (sent) {
            for (int k = 0; k <= t->neighbour_count[target]; ++k) {
                const int p = x->send_entry[k];
                if (p < 0) continue;
                const int mc = t->source_atom[destination][p];
                const int neighbour = t->source_neighbour[destination][p];
                const int orbitals = OpenMX_VnaCase2Orbitals(t, t->neighbours[target][k]);
                const size_t values = (size_t)orbitals * x->projectors;
                for (int d = 0; d < 4; ++d) {
                    float *dest = shift ? wire : openmx_float_block_row(projectors, d, slot, k, 0);
                    memcpy(dest, openmx_float_block_row(projectors, d, mc, neighbour, 0),
                        values * sizeof(*dest));
                    if (shift) wire += values;
                }
            }
        }
        if (shift) {
            OpenMX_VnaCase2SendRecv(x, send, sent, destination, receive, received, source, x->datatype);
            wire = receive;
            for (int k = x->receive_head[source]; received && k >= 0; k = x->receive_next[k]) {
                const int orbitals = OpenMX_VnaCase2Orbitals(t, t->neighbours[centre][k]);
                const size_t values = (size_t)orbitals * x->projectors;
                for (int d = 0; d < 4; ++d) {
                    memcpy(openmx_float_block_row(projectors, d, slot, k, 0), wire,
                        values * sizeof(*wire));
                    wire += values;
                }
            }
        }
    }
}

#endif
