#include "inc/a.h"
#define HDR "inc/b.h"
#include HDR
#define ANGLE <inc/b.h>
#include ANGLE
#include "inc/once.h"
#include "inc/once.h"
#include "inc/guard.h"
#include "inc/guard.h"
after_includes A_VALUE B_VALUE ONCE_VALUE GUARD_VALUE __LINE__
#if __has_include("inc/a.h") && !__has_include("inc/nonexistent.h")
has_include_ok
#endif
