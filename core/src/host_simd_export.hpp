#ifndef OPENMX_HOST_SIMD_EXPORT_HPP
#define OPENMX_HOST_SIMD_EXPORT_HPP
// The private host SIMD library exports only POD dispatch tables and plain
// integers through C entry points. Everything else keeps hidden visibility.
#if defined(_WIN32)
#ifdef OPENMX_HOST_SIMD_BUILD
#define OPENMX_HOST_SIMD_EXPORT __declspec(dllexport)
#else
#define OPENMX_HOST_SIMD_EXPORT __declspec(dllimport)
#endif
#else
#define OPENMX_HOST_SIMD_EXPORT __attribute__((visibility("default")))
#endif
#endif
