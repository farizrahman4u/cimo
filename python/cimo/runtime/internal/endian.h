/* cimo runtime: arm64 is little-endian (Linux and macOS). */
#ifndef __CIMO_RT_ENDIAN_H
#define __CIMO_RT_ENDIAN_H
#define __LITTLE_ENDIAN 1234
#define __BIG_ENDIAN 4321
#define __BYTE_ORDER __LITTLE_ENDIAN
#define LITTLE_ENDIAN __LITTLE_ENDIAN
#define BIG_ENDIAN __BIG_ENDIAN
#define BYTE_ORDER __BYTE_ORDER
#endif
