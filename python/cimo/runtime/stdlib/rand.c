/* musl src/prng/srand.c and rand.c. One generator per module, shared by its threads (C does not
   require rand to be thread-safe). */
#include <stdlib.h>
#include <stdint.h>

static uint64_t seed;

void srand(unsigned s)
{
	seed = s-1;
}

int rand(void)
{
	seed = 6364136223846793005ULL*seed + 1;
	return seed>>33;
}
