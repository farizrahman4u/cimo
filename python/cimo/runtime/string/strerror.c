/* cimo: the runtime has no errno and no OS, so only the error numbers C11 defines (EDOM,
   ERANGE, EILSEQ, with the host's values) get a message; messages as in musl. */
#include <string.h>

char *strerror(int e)
{
#ifdef __APPLE__
	enum { EDOM_ = 33, ERANGE_ = 34, EILSEQ_ = 92 };
#else
	enum { EDOM_ = 33, ERANGE_ = 34, EILSEQ_ = 84 };
#endif
	switch (e) {
	case EDOM_: return "Domain error";
	case ERANGE_: return "Result not representable";
	case EILSEQ_: return "Illegal byte sequence";
	}
	return "No error information";
}
