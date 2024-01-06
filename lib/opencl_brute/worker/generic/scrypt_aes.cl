/*
    Scrypt OpenCL Optimized kernel
    (c) C.B. and B. Kerler 2018-2019
    MIT License
    
    AES-256-CBC decrypting code by Omegaice
*/ 

// [Lines 1 and 2 are for defining N and invMemoryDensity, and must be blank]

/*
sCrypt kernel.. or just ROMix really, for use with my sBrute PyOpenCL core
Originally adapted from Bjorn Kerler's opencl_brute

Follows the variable names of wikipedia's psuedocode:
    https://en.wikipedia.org/wiki/Scrypt#Algorithm
Function/macro convention is F(output, input_1, input_2, ..), i.e. output first.
Generally work with pointers.
 
=== Design choices & reasoning =================================================

> initial and final pbkdf2s are left to python for a few reasons:
    - vastly simplier cl code, hopefully giving us better optimisation
    - reduced bugs
    - simplier parallelisation across the parameter 'p'
    - not a burden on python: work is tiny..
        & the special sBrute python core is careful that any work is while the GPUs are busy

> salsa20 is sort of inplace
    - fundamentally needs to copy the input internally
    - does (hopefully) make savings by having input = output, making the algo:
        orig_input < input
        Process(input)      // inplace
        input ^= orig_input
      where the last line should be faster than output = input ^ orig_input

> JUMBLES!
    - jumble(Y0|Y1|..|Y_2r-1) = Y0|Y2|..|Y_2r-1  |  Y1|Y3|..|Y_2r-1,
        which is effectively performed at the end of BlockMix in the original definition
    - jumble is of order 4, i.e. jumble^4 = id
    - we want to avoid doing this copying..
    - naturally we unroll the loop in BlockMix, so reordering the input is free
=> all this leads to us working in 4 different states of "jumbled-ness" throughout the program
    - indeed our V[j]s are jumbled j % 4 times.
    - xoring the V[j]'s back onto a (somewhat jumbled) X in the 2nd loop effectively requires a function call

> Salsa function is long, so can't be macro-ed and called lots of times.
    - We could have kept the BlockMix loop,
        but this would require reading the jumble index from an array each iteration
    - Instead we make Salsa a void Function
    - Also a xor loop is moved into Salsa, so that we can unroll it,
      at the small cost of an extra parameter

> All values except our huge V array are kept locally.
    - V[j] is accessed and xored onto a local array.

> After a long battle, the Salsa20/8's 4-pairs-of-rounds loop is unrolled.
    - Program size should still be fine.

> using "= {0}" to initialise local arrays is the classic fix copied from Bjorn Kerler's code:
    seems to be necessary to actually make the program work, even though it should have no effect.


=== FIN ========================================================================
*/




// ===========================================================================
// 1 / memory density
#ifndef invMemoryDensity
    #define invMemoryDensity 1
#endif
#define iMD_is_pow_2 (!(invMemoryDensity & (invMemoryDensity - 1)) && invMemoryDensity)


// sCrypt constants :
//  - p irrelevant to us
//  - r below cannot be changed (without altering the program)
//      > makes the 'jumble' operation order 4
//  - N can be changed if necessary, up until we run out of buffer space (so maybe <= 20?)
#ifndef N
    #define N 15        // <= 20?
#endif


#define r 8         // CAN'T BE CHANGED

// derivatives of constants :s
#define blockSize_bytes (128 * r)   // 1024
#define ceilDiv(n,d) (((n) + (d) - 1) / (d))
#define blockSize_int32 ceilDiv(blockSize_bytes, 4) // 256
#define iterations (1 << N) 

// Useful struct for internal processing: a lump of 64 bytes (sort of an atomic unit)
typedef struct {
    unsigned int buffer[16];    // 64 bytes
} T_Lump64;

// Comfy Block struct
typedef struct {
	T_Lump64 lump[2*r];    // 1024 bytes
} T_Block;

// Struct for the large V array which needs to be pseduo-randomly accessed.
// Now restricted in length by invMemoryDensity
typedef struct {
    T_Block blk[ceilDiv(iterations, invMemoryDensity)];
} T_HugeArray;






// ===========================================================================
// Simple macros
// Lump & Block macros take pointers

#define copy16_unrolled(dest,src)                    \
/* dest[i] = src[i] for i in [0..16) */     \
{                       \
    dest[0]  = src[0];  \
    dest[1]  = src[1];  \
    dest[2]  = src[2];  \
    dest[3]  = src[3];  \
    dest[4]  = src[4];  \
    dest[5]  = src[5];  \
    dest[6]  = src[6];  \
    dest[7]  = src[7];  \
    dest[8]  = src[8];  \
    dest[9]  = src[9];  \
    dest[10] = src[10]; \
    dest[11] = src[11]; \
    dest[12] = src[12]; \
    dest[13] = src[13]; \
    dest[14] = src[14]; \
    dest[15] = src[15]; \
}

#define xor16_unrolled(dest,src)            \
/* dest[i] ^= src[i] for i in [0..16) */    \
{                        \
    dest[0]  ^= src[0];  \
    dest[1]  ^= src[1];  \
    dest[2]  ^= src[2];  \
    dest[3]  ^= src[3];  \
    dest[4]  ^= src[4];  \
    dest[5]  ^= src[5];  \
    dest[6]  ^= src[6];  \
    dest[7]  ^= src[7];  \
    dest[8]  ^= src[8];  \
    dest[9]  ^= src[9];  \
    dest[10] ^= src[10]; \
    dest[11] ^= src[11]; \
    dest[12] ^= src[12]; \
    dest[13] ^= src[13]; \
    dest[14] ^= src[14]; \
    dest[15] ^= src[15]; \
}

#define add16_unrolled(dest, src)   \
/* dest[i] += src[i] for i in [0..16) */    \
{                                   \
    dest[0] += src[0];  \
    dest[1] += src[1];  \
    dest[2] += src[2];  \
    dest[3] += src[3];  \
    dest[4] += src[4];  \
    dest[5] += src[5];  \
    dest[6] += src[6];  \
    dest[7] += src[7];  \
    dest[8] += src[8];  \
    dest[9] += src[9];  \
    dest[10] += src[10];    \
    dest[11] += src[11];    \
    dest[12] += src[12];    \
    dest[13] += src[13];    \
    dest[14] += src[14];    \
    dest[15] += src[15];    \
}

#define copyLump64_unrolled(dest, src)  \
/* &dest = &src */                        \
{                                       \
    copy16_unrolled(dest->buffer, src->buffer)  \
}

#define xorLump64_unrolled(dest, src)   \
/* &dest ^= &src */                       \
{                                       \
    xor16_unrolled(dest->buffer, src->buffer)   \
}

#define copyBlock_halfrolled(destTag, dest, srcTag, src)     \
/* [destTag] &dest = [srcTag] &src, copying lumps of 64 in a loop */ \
{                                           \
    destTag T_Lump64* _CB_d;                \
    srcTag T_Lump64* _CB_s;                 \
    for (int i = 2*r - 1; i >= 0; i--)      \
    {                                       \
        _CB_d = &(dest)->lump[i];           \
        _CB_s = &(src)->lump[i];            \
        copyLump64_unrolled(_CB_d, _CB_s)   \
    }                                       \
}

#define xorBlock_halfrolled(destTag, dest, srcTag, src)     \
/* [destTag] &dest ^= [srcTag] &src, xoring lumps of 64 in a loop */ \
{                                           \
    destTag T_Lump64* _XB_d;                \
    srcTag T_Lump64* _XB_s;                 \
    for (int i = 2*r - 1; i >= 0; i--)      \
    {                                       \
        _XB_d = &(dest)->lump[i];           \
        _XB_s = &(src)->lump[i];            \
        xorLump64_unrolled(_XB_d, _XB_s)    \
    }                                       \
}







// ==========================================================================
// Debug printing macros

#define printLump(lump) \
/* Takes the object not a pointer */    \
{                                       \
    for (int j = 0; j < 16; j++){       \
        printf("%08X", lump.buffer[j]); \
    }                                   \
}

#define printBlock(blk) \
/* Takes a pointer */   \
{                                   \
    for (int i = 0; i < 2*r; i++)   \
    {                               \
        printLump(blk->lump[i])     \
    }                               \
}







// ===========================================================================
// Salsa 20/8
// Adapted from https://en.wikipedia.org/wiki/Salsa20#Structure


// Rotation synonym and quarter round for Salsa20
#define rotl32(a,n) rotate((a), (n))
#define quarterRound(a, b, c, d)		\
/**/                                    \
{                                       \
	b ^= rotl32(a + d,  7u);	        \
	c ^= rotl32(b + a,  9u);	        \
	d ^= rotl32(c + b, 13u);	        \
	a ^= rotl32(d + c, 18u);            \
}

#define pairOfRounds(x)                         \
/* Pinched from wikipedia */                    \
{                                               \
    /* Odd round */                             \
    quarterRound(x[ 0], x[ 4], x[ 8], x[12]);   \
    quarterRound(x[ 5], x[ 9], x[13], x[ 1]);	\
    quarterRound(x[10], x[14], x[ 2], x[ 6]);	\
    quarterRound(x[15], x[ 3], x[ 7], x[11]);	\
    /* Even round */                            \
    quarterRound(x[ 0], x[ 1], x[ 2], x[ 3]);	\
    quarterRound(x[ 5], x[ 6], x[ 7], x[ 4]);	\
    quarterRound(x[10], x[11], x[ 8], x[ 9]);	\
    quarterRound(x[15], x[12], x[13], x[14]);	\
}

// Function not a macro (see 'design choices' at the top)
// Xors X onto lump then computes lump <- Salsa20/8(lump)
__private void Xor_then_Salsa_20_8_InPlace(__private T_Lump64* lump, __private T_Lump64* X)
{
    // Includes xoring here, to allow for unrolling (at expense of an extra param)
    xorLump64_unrolled(lump, X)

    // Copy input into x (lowercase) for processing
    unsigned int x[16] = {0};
    copy16_unrolled(x, lump->buffer)

    // Do the 8 rounds
    // After much internal conflict I have unrolled this loop of 4
    pairOfRounds(x)
    pairOfRounds(x)
    pairOfRounds(x)
    pairOfRounds(x)

    // Add x to original input, and store into output.. which is the input :)
    add16_unrolled(lump->buffer, x)
}







// ====================================================================================
// BlockMix variants
//   Nomenclature of the variants is composition: f_g_h(x) = f(g(h(x)))


#define BlockMixLoopBody(_B_i, _BMLB_X)      \
/* My heavily adapted BlockMix loop body */ \
{                                           \
    /*  _B_i = _B_i ^ _BMLB_X
        _B_i = Salsa20(_B_i)
        _BMLB_X = _B_i (as pointers)
        [ Doesn't increment i ]
    */                                        \
    Xor_then_Salsa_20_8_InPlace(_B_i, _BMLB_X);\
    _BMLB_X = _B_i;                            \
}

#define _BlockMix_Generic(B, \
                        i_1, i_2, i_3, i_4, i_5, i_6, i_7,         \
                        i_8, i_9, i_10, i_11, i_12, i_13, i_14, i_15)   \
/* Takes {i_0, .. , i_15} a permutation of {0, .. , 15}, the order of indices
    i_0 = 0 implied. */                                                 \
{                                                                       \
    /* Don't even need to copy to _BM_X, can just point! */                 \
    /* Start with _BM_X = B[2r-1] (indexing across blocks of 64 bytes) */   \
    __private T_Lump64* _BM_X = &B->lump[i_15];   \
    __private T_Lump64* _BM_B_i;                  \
                                        \
    /* i_0 = 0 */                       \
    BlockMixLoopBody(&B->lump[0], _BM_X)\
    _BM_B_i = &B->lump[i_1];            \
    BlockMixLoopBody(_BM_B_i, _BM_X)    \
    _BM_B_i = &B->lump[i_2];            \
    BlockMixLoopBody(_BM_B_i, _BM_X)    \
    _BM_B_i = &B->lump[i_3];            \
    BlockMixLoopBody(_BM_B_i, _BM_X)    \
                                    \
    _BM_B_i = &B->lump[i_4];            \
    BlockMixLoopBody(_BM_B_i, _BM_X)    \
    _BM_B_i = &B->lump[i_5];            \
    BlockMixLoopBody(_BM_B_i, _BM_X)    \
    _BM_B_i = &B->lump[i_6];            \
    BlockMixLoopBody(_BM_B_i, _BM_X)    \
    _BM_B_i = &B->lump[i_7];            \
    BlockMixLoopBody(_BM_B_i, _BM_X)    \
                                    \
    _BM_B_i = &B->lump[i_8];            \
    BlockMixLoopBody(_BM_B_i, _BM_X)    \
    _BM_B_i = &B->lump[i_9];            \
    BlockMixLoopBody(_BM_B_i, _BM_X)    \
    _BM_B_i = &B->lump[i_10];           \
    BlockMixLoopBody(_BM_B_i, _BM_X)    \
    _BM_B_i = &B->lump[i_11];           \
    BlockMixLoopBody(_BM_B_i, _BM_X)    \
                                    \
    _BM_B_i = &B->lump[i_12];           \
    BlockMixLoopBody(_BM_B_i, _BM_X)    \
    _BM_B_i = &B->lump[i_13];           \
    BlockMixLoopBody(_BM_B_i, _BM_X)    \
    _BM_B_i = &B->lump[i_14];           \
    BlockMixLoopBody(_BM_B_i, _BM_X)    \
    _BM_B_i = &B->lump[i_15];           \
    BlockMixLoopBody(_BM_B_i, _BM_X)    \
}


#define BlockMix_J3(B) \
/* 3 jumbles then a BlockMix */   \
{    \
    _BlockMix_Generic(B, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15)  \
}

#define J1_BlockMix_J2(B) \
/* Jumble twice, BlockMixes, then jumbles.  */   \
{    \
    _BlockMix_Generic(B, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15)  \
}

#define J2_BlockMix_J1(B) \
/* Jumbles, BlockMixes, then 2 jumbles. */   \
{    \
    _BlockMix_Generic(B, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15)  \
}

#define J3_BlockMix(B) \
/* BlockMix followed by 3 jumbles (i.e. a jumble-inverse) */   \
{    \
    _BlockMix_Generic(B, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)  \
}








// ===============================================================================
// Integerify: gets it's own section

#define Integerify(j, block)                    \
/* Observe that the last 64 bytes is the last lump */ \
/* Correct regardless of the jumbled-ness of the block! */ \
/* Requires N <= 32 */ \
{                                               \
    j = block->lump[15].buffer[0] % iterations; \
}






// ===============================================================================
// Xoring methods for the 4 states of jumbled-ness
//   Culminates in the 'recover_and_xor_appropriately' function, which selects the correct one.

#define _xor_generic(dest, srcTag, src,                \
        i_0, i_1, i_2, i_3, i_4, i_5, i_6, i_7,         \
        i_8, i_9, i_10, i_11, i_12, i_13, i_14, i_15)   \
/* dest ^= perm(src), xor permuted source on, k -> i_k the permutation.
    requires src disjoint from dest : guaranteed by address spaces */      \
{                                           \
    __private T_Lump64* _XB_d;              \
    srcTag T_Lump64* _XB_s;                 \
    const int perm[16] = {i_0, i_1, i_2, i_3, i_4, i_5, i_6, i_7,   \
                    i_8, i_9, i_10, i_11, i_12, i_13, i_14, i_15};  \
    for (int i = 2*r - 1; i >= 0; i--)      \
    {                                       \
        _XB_d = &(dest)->lump[i];           \
        /* Select perm index instead of index */    \
        _XB_s = &(src)->lump[perm[i]];      \
        xorLump64_unrolled(_XB_d, _XB_s)    \
    }                                       \
}

#define xor_J1(dest, srcTag, src)   \
{                           \
    _xor_generic(dest, srcTag, src, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15)   \
}

#define xor_J2(dest, srcTag, src)   \
{                           \
    _xor_generic(dest, srcTag, src, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15)   \
}

#define xor_J3(dest, srcTag, src)   \
{                           \
    _xor_generic(dest, srcTag, src, 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15)   \
}

// Chooses the appropriate xoring based on the supplied value diff, which is modded by 4
//   diff is such that jumble^diff(inp) is 'equally jumbled' as out
//   diff will be pseudorandom, so case statement should maximise efficiency.
// Now also recomputes V'[j] from V[j // density]
void recover_and_xor_appropriately(__private T_Block* dest, __global T_Block* V, 
        unsigned int j, unsigned int diff){

    // Number of computations to make.
    int nComps = j % invMemoryDensity;
    int V_index = j / invMemoryDensity;

    if (nComps == 0){
        label_nComps_is_zero:
        // Do the xoring directly from the global block V[V_index]
        // Basically the old "xor_appropriately"
        switch(diff % 4){
            case 0:
                xorBlock_halfrolled(__private, dest, __global, &V[V_index])
                break;
            case 1:
                xor_J1(dest, __global, &V[V_index])
                break;
            case 2:
                xor_J2(dest, __global, &V[V_index])
                break;
            case 3:
                xor_J3(dest, __global, &V[V_index])
                break;
        }
    }
    else
    {
        // Copy V[j/iMD] into Y, where we'll do our work
        //   (using Bjorn's initialisation-bug-prevention once more)
        // Observe that this copy is pretty essential
        __private unsigned int _Y_bytes[ceilDiv(sizeof(T_Block), 4)] = {0};
        __private T_Block* Y = (T_Block*) _Y_bytes;
        copyBlock_halfrolled(__private, Y, __global, &V[V_index])

        // We have to decide where to enter the loop, based on how jumbled V[V_index] is
        //   i.e. (V_index * invMemoryDensity) % 4
        switch((j - nComps) % 4){
            case 0:
                goto label_j0;
            case 1:
                goto label_j3;
            case 2:
                goto label_j2;
            case 3:
                goto label_j1;
        }

        // Could change to nComps-- .. would save an assembly instruction? :)
        do {
            label_j0: J3_BlockMix(Y);
            if (--nComps == 0){
                break;
            }

            label_j3: J2_BlockMix_J1(Y);
            if (--nComps == 0){
                break;
            }

            label_j2: J1_BlockMix_J2(Y);
            if (--nComps == 0){
                break;
            }

            label_j1: BlockMix_J3(Y);
        } while (--nComps > 0);


        // With Y = V'[j] recovered, we can finish the job off by xoring appropriately.
        switch(diff % 4){
            case 0:
                xorBlock_halfrolled(__private, dest, __private, Y)
                break;
            case 1:
                xor_J1(dest, __private, Y)
                break;
            case 2:
                xor_J2(dest, __private, Y)
                break;
            case 3:
                xor_J3(dest, __private, Y)
                break;
        }
    }

}









// ==================================================================================
// The big one: ROMix kernel

__kernel void ROMix( __global T_Block* blocksFlat,
                    __global T_HugeArray* hugeArraysFlat,
                    __global T_Block* outputsFlat
                    )
{
    // Get our id and so unflatten our block & huge array 'V', to get pointers
    //   &arr[i] and arr + i should be equivalent syntax?
    __private unsigned int id = get_global_id(0);
    __global T_Block* origBlock = &blocksFlat[id];
    __global T_Block* outputBlock = &outputsFlat[id];
    __global T_Block* V = hugeArraysFlat[id].blk;
    __global T_Block* curr_V_blk = V;
    
    // Copy our block into local X : could roll fully
    //   slightly weird to allow for Bjorn's bug-preventing-initialisation
    __private unsigned int _X_bytes[ceilDiv(sizeof(T_Block), 4)] = {0};
    __private T_Block* X = (T_Block*) _X_bytes;
    copyBlock_halfrolled(__private, X, __global, origBlock)



    // =====================================================
    // 1st loop, fill V with the correct values, in varying states of jumbled-ness:
    //  Let V' be the correct value. d the invMemoryDensity
    //  d*i mod 4     ||      state in V[i]
    // ============================================
    //      0         ||          V'[d*i]
    //      1         ||      J^3(V'[d*i])
    //      2         ||      J^2(V'[d*i])
    //      3         ||      J^1(V'[d*i])    
    // Now only storing the first in every invMemoryDensity

    #define maybeStore(curr_V_blk, X, _j)   \
    /* If due, stores X to curr_V_blk and increments it */  \
    {                                       \
        if ((_j) % invMemoryDensity == 0){  \
            copyBlock_halfrolled(__global, curr_V_blk, __private, X);   \
            curr_V_blk++;                   \
        }                                   \
    }

    // Still needs to do all 'iterations' loops, to compute the final X
    for (int j = 0; j < iterations; j+=4){
        maybeStore(curr_V_blk, X, j)
        J3_BlockMix(X);

        maybeStore(curr_V_blk, X, j+1)
        J2_BlockMix_J1(X);

        maybeStore(curr_V_blk, X, j+2)
        J1_BlockMix_J2(X);

        maybeStore(curr_V_blk, X, j+3)
        BlockMix_J3(X);
    }

    #undef maybeStore


    // ====================================================
    // 2nd loop, similarly X passes through 4 states of jumbled-ness
    // Observe that we need to choose our xor based on j-i % 4,
    //   which adds more complexity compared to the first loop.

    // Moreover we may need to actually recompute the value.
    // => sensibly (in terms of program length) this is in "recover_and_xor_appropriately"
    unsigned int j;
    for (unsigned int i = 0; i < iterations; i+=4){
        Integerify(j, X)
        recover_and_xor_appropriately(X, V, j, j - i);
        J3_BlockMix(X);

        Integerify(j, X);
        recover_and_xor_appropriately(X, V, j, j - (i+1));
        J2_BlockMix_J1(X);

        Integerify(j, X);
        recover_and_xor_appropriately(X, V, j, j - (i+2));
        J1_BlockMix_J2(X);

        Integerify(j, X);
        recover_and_xor_appropriately(X, V, j, j - (i+3));
        BlockMix_J3(X);
    }

    // Copy to output: could roll fully
    copyBlock_halfrolled(__global, outputBlock, __private, X)
}






// ===============================================================================
// For testing, Salsa20's each lump in place
// Same signature as ROMix for ease
__kernel void Salsa20(  __global T_Block* blocksFlat,
                        __global T_HugeArray* hugeArraysFlat,
                        __global T_Block* outputsFlat)
{
    __private unsigned int id = get_global_id(0);

    // Copy locally, initialising first for fear of bugs
    __private unsigned int _b[ceilDiv(sizeof(T_Block), 4)] = {0};
    __private T_Block* blk = (T_Block*) _b;
    copyBlock_halfrolled(__private, blk, __global, (&blocksFlat[id]))

    // Initialise a zero lump
    unsigned int _z[ceilDiv(sizeof(T_Lump64), 4)] = {0};
    T_Lump64* zeroLump = (T_Lump64*)_z;
    
    // Salsa each lump inPlace
    for (int j = 0; j < 2*r; j++)
    {
        Xor_then_Salsa_20_8_InPlace((&blk->lump[j]), zeroLump);
    }

    // Copy to output
    __global T_Block* output = &outputsFlat[id];
    copyBlock_halfrolled(__global, output, __private, blk)
}



#### AES_PART_TODO
/*
    PBKDF2 SHA1 OpenCL Optimized kernel, limited to max. 32 chars for salt and password
    (c) B. Kerler 2017
    MIT License
*/
/*
    AES roundkeys computing code by adrianbelgun
*/
/*
    AES-256-CBC decrypting code by Omegaice
*/


//PBKDF2 Related
#define CONST_BYT_ACTUAL_PWLEN 7 //actual password length.
#define CONST_BYT_SALTLEN 16 //should always be 16, according to PBKDF2 standard
#define PBKDFITER 4000//significantly affects the speed(almost linear), however, wrong iter value won't harvest expected correct result.

//AES Related
#define AES_CBC_256_KEY_BYTE_SIZE 32//AES-256-CBC uses a key of 256bit=32byte, this will also significantly affect the speed(longer->slower)
#define ENC_DATA_BLOCKSIZE 16 //should be always 16, according to the AES standard 


//----------------------------------------------PBKDF2-SHA1-------------------------------------------------------

#define rotl32(a,n) rotate ((a), (n)) 

uint SWAP (uint val)
{
    return (rotate(((val) & 0x00FF00FF), 24U) | rotate(((val) & 0xFF00FF00), 8U));
}

#define mod(x,y) x-(x/y*y)

#define F2(x,y,z)  ((x) ^ (y) ^ (z))
#define F1(x,y,z)   (bitselect(z,y,x))
#define F0(x,y,z)   (bitselect (x, y, (x ^ z)))

#define SHA1M_A 0x67452301u
#define SHA1M_B 0xefcdab89u
#define SHA1M_C 0x98badcfeu
#define SHA1M_D 0x10325476u
#define SHA1M_E 0xc3d2e1f0u

#define SHA1C00 0x5a827999u
#define SHA1C01 0x6ed9eba1u
#define SHA1C02 0x8f1bbcdcu
#define SHA1C03 0xca62c1d6u

#define SHA1_STEP(f,a,b,c,d,e,x)    \
{                                   \
  e += K;                           \
  e += x;                           \
  e += f (b, c, d);                 \
  e += rotl32 (a,  5u);             \
  b  = rotl32 (b, 30u);             \
}

static void sha1_process2 (const uint *W, uint *digest)
{
  uint A = digest[0];
  uint B = digest[1];
  uint C = digest[2];
  uint D = digest[3];
  uint E = digest[4];

  uint w0_t = W[0];
  uint w1_t = W[1];
  uint w2_t = W[2];
  uint w3_t = W[3];
  uint w4_t = W[4];
  uint w5_t = W[5];
  uint w6_t = W[6];
  uint w7_t = W[7];
  uint w8_t = W[8];
  uint w9_t = W[9];
  uint wa_t = W[10];
  uint wb_t = W[11];
  uint wc_t = W[12];
  uint wd_t = W[13];
  uint we_t = W[14];
  uint wf_t = W[15];

  #undef K
  #define K SHA1C00

  SHA1_STEP (F1, A, B, C, D, E, w0_t);
  SHA1_STEP (F1, E, A, B, C, D, w1_t);
  SHA1_STEP (F1, D, E, A, B, C, w2_t);
  SHA1_STEP (F1, C, D, E, A, B, w3_t);
  SHA1_STEP (F1, B, C, D, E, A, w4_t);
  SHA1_STEP (F1, A, B, C, D, E, w5_t);
  SHA1_STEP (F1, E, A, B, C, D, w6_t);
  SHA1_STEP (F1, D, E, A, B, C, w7_t);
  SHA1_STEP (F1, C, D, E, A, B, w8_t);
  SHA1_STEP (F1, B, C, D, E, A, w9_t);
  SHA1_STEP (F1, A, B, C, D, E, wa_t);
  SHA1_STEP (F1, E, A, B, C, D, wb_t);
  SHA1_STEP (F1, D, E, A, B, C, wc_t);
  SHA1_STEP (F1, C, D, E, A, B, wd_t);
  SHA1_STEP (F1, B, C, D, E, A, we_t);
  SHA1_STEP (F1, A, B, C, D, E, wf_t);
  w0_t = rotl32 ((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u); SHA1_STEP (F1, E, A, B, C, D, w0_t);
  w1_t = rotl32 ((we_t ^ w9_t ^ w3_t ^ w1_t), 1u); SHA1_STEP (F1, D, E, A, B, C, w1_t);
  w2_t = rotl32 ((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u); SHA1_STEP (F1, C, D, E, A, B, w2_t);
  w3_t = rotl32 ((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u); SHA1_STEP (F1, B, C, D, E, A, w3_t);

  #undef K
  #define K SHA1C01

  w4_t = rotl32 ((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u); SHA1_STEP (F2, A, B, C, D, E, w4_t);
  w5_t = rotl32 ((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u); SHA1_STEP (F2, E, A, B, C, D, w5_t);
  w6_t = rotl32 ((w3_t ^ we_t ^ w8_t ^ w6_t), 1u); SHA1_STEP (F2, D, E, A, B, C, w6_t);
  w7_t = rotl32 ((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u); SHA1_STEP (F2, C, D, E, A, B, w7_t);
  w8_t = rotl32 ((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u); SHA1_STEP (F2, B, C, D, E, A, w8_t);
  w9_t = rotl32 ((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u); SHA1_STEP (F2, A, B, C, D, E, w9_t);
  wa_t = rotl32 ((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u); SHA1_STEP (F2, E, A, B, C, D, wa_t);
  wb_t = rotl32 ((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u); SHA1_STEP (F2, D, E, A, B, C, wb_t);
  wc_t = rotl32 ((w9_t ^ w4_t ^ we_t ^ wc_t), 1u); SHA1_STEP (F2, C, D, E, A, B, wc_t);
  wd_t = rotl32 ((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u); SHA1_STEP (F2, B, C, D, E, A, wd_t);
  we_t = rotl32 ((wb_t ^ w6_t ^ w0_t ^ we_t), 1u); SHA1_STEP (F2, A, B, C, D, E, we_t);
  wf_t = rotl32 ((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u); SHA1_STEP (F2, E, A, B, C, D, wf_t);
  w0_t = rotl32 ((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u); SHA1_STEP (F2, D, E, A, B, C, w0_t);
  w1_t = rotl32 ((we_t ^ w9_t ^ w3_t ^ w1_t), 1u); SHA1_STEP (F2, C, D, E, A, B, w1_t);
  w2_t = rotl32 ((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u); SHA1_STEP (F2, B, C, D, E, A, w2_t);
  w3_t = rotl32 ((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u); SHA1_STEP (F2, A, B, C, D, E, w3_t);
  w4_t = rotl32 ((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u); SHA1_STEP (F2, E, A, B, C, D, w4_t);
  w5_t = rotl32 ((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u); SHA1_STEP (F2, D, E, A, B, C, w5_t);
  w6_t = rotl32 ((w3_t ^ we_t ^ w8_t ^ w6_t), 1u); SHA1_STEP (F2, C, D, E, A, B, w6_t);
  w7_t = rotl32 ((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u); SHA1_STEP (F2, B, C, D, E, A, w7_t);

  #undef K
  #define K SHA1C02

  w8_t = rotl32 ((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u); SHA1_STEP (F0, A, B, C, D, E, w8_t);
  w9_t = rotl32 ((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u); SHA1_STEP (F0, E, A, B, C, D, w9_t);
  wa_t = rotl32 ((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u); SHA1_STEP (F0, D, E, A, B, C, wa_t);
  wb_t = rotl32 ((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u); SHA1_STEP (F0, C, D, E, A, B, wb_t);
  wc_t = rotl32 ((w9_t ^ w4_t ^ we_t ^ wc_t), 1u); SHA1_STEP (F0, B, C, D, E, A, wc_t);
  wd_t = rotl32 ((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u); SHA1_STEP (F0, A, B, C, D, E, wd_t);
  we_t = rotl32 ((wb_t ^ w6_t ^ w0_t ^ we_t), 1u); SHA1_STEP (F0, E, A, B, C, D, we_t);
  wf_t = rotl32 ((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u); SHA1_STEP (F0, D, E, A, B, C, wf_t);
  w0_t = rotl32 ((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u); SHA1_STEP (F0, C, D, E, A, B, w0_t);
  w1_t = rotl32 ((we_t ^ w9_t ^ w3_t ^ w1_t), 1u); SHA1_STEP (F0, B, C, D, E, A, w1_t);
  w2_t = rotl32 ((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u); SHA1_STEP (F0, A, B, C, D, E, w2_t);
  w3_t = rotl32 ((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u); SHA1_STEP (F0, E, A, B, C, D, w3_t);
  w4_t = rotl32 ((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u); SHA1_STEP (F0, D, E, A, B, C, w4_t);
  w5_t = rotl32 ((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u); SHA1_STEP (F0, C, D, E, A, B, w5_t);
  w6_t = rotl32 ((w3_t ^ we_t ^ w8_t ^ w6_t), 1u); SHA1_STEP (F0, B, C, D, E, A, w6_t);
  w7_t = rotl32 ((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u); SHA1_STEP (F0, A, B, C, D, E, w7_t);
  w8_t = rotl32 ((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u); SHA1_STEP (F0, E, A, B, C, D, w8_t);
  w9_t = rotl32 ((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u); SHA1_STEP (F0, D, E, A, B, C, w9_t);
  wa_t = rotl32 ((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u); SHA1_STEP (F0, C, D, E, A, B, wa_t);
  wb_t = rotl32 ((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u); SHA1_STEP (F0, B, C, D, E, A, wb_t);

  #undef K
  #define K SHA1C03

  wc_t = rotl32 ((w9_t ^ w4_t ^ we_t ^ wc_t), 1u); SHA1_STEP (F2, A, B, C, D, E, wc_t);
  wd_t = rotl32 ((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u); SHA1_STEP (F2, E, A, B, C, D, wd_t);
  we_t = rotl32 ((wb_t ^ w6_t ^ w0_t ^ we_t), 1u); SHA1_STEP (F2, D, E, A, B, C, we_t);
  wf_t = rotl32 ((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u); SHA1_STEP (F2, C, D, E, A, B, wf_t);
  w0_t = rotl32 ((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u); SHA1_STEP (F2, B, C, D, E, A, w0_t);
  w1_t = rotl32 ((we_t ^ w9_t ^ w3_t ^ w1_t), 1u); SHA1_STEP (F2, A, B, C, D, E, w1_t);
  w2_t = rotl32 ((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u); SHA1_STEP (F2, E, A, B, C, D, w2_t);
  w3_t = rotl32 ((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u); SHA1_STEP (F2, D, E, A, B, C, w3_t);
  w4_t = rotl32 ((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u); SHA1_STEP (F2, C, D, E, A, B, w4_t);
  w5_t = rotl32 ((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u); SHA1_STEP (F2, B, C, D, E, A, w5_t);
  w6_t = rotl32 ((w3_t ^ we_t ^ w8_t ^ w6_t), 1u); SHA1_STEP (F2, A, B, C, D, E, w6_t);
  w7_t = rotl32 ((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u); SHA1_STEP (F2, E, A, B, C, D, w7_t);
  w8_t = rotl32 ((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u); SHA1_STEP (F2, D, E, A, B, C, w8_t);
  w9_t = rotl32 ((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u); SHA1_STEP (F2, C, D, E, A, B, w9_t);
  wa_t = rotl32 ((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u); SHA1_STEP (F2, B, C, D, E, A, wa_t);
  wb_t = rotl32 ((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u); SHA1_STEP (F2, A, B, C, D, E, wb_t);
  wc_t = rotl32 ((w9_t ^ w4_t ^ we_t ^ wc_t), 1u); SHA1_STEP (F2, E, A, B, C, D, wc_t);
  wd_t = rotl32 ((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u); SHA1_STEP (F2, D, E, A, B, C, wd_t);
  we_t = rotl32 ((wb_t ^ w6_t ^ w0_t ^ we_t), 1u); SHA1_STEP (F2, C, D, E, A, B, we_t);
  wf_t = rotl32 ((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u); SHA1_STEP (F2, B, C, D, E, A, wf_t);

  digest[0] += A;
  digest[1] += B;
  digest[2] += C;
  digest[3] += D;
  digest[4] += E;
} 

static void pbkdf(const uint *pass, int pass_len,__constant const uint *salt, int salt_len,uint* hash)
{
    int plen=pass_len/4;
    if (mod(pass_len,4)) plen++;

    int slen=salt_len/4;
    if (mod(salt_len,4)) slen++;

    uint* p = hash;

    uint ipad[16];
    ipad[0x0]=0x36363636;
    ipad[0x1]=0x36363636;
    ipad[0x2]=0x36363636;
    ipad[0x3]=0x36363636;
    ipad[0x4]=0x36363636;
    ipad[0x5]=0x36363636;
    ipad[0x6]=0x36363636;
    ipad[0x7]=0x36363636;
    ipad[0x8]=0x36363636;
    ipad[0x9]=0x36363636;
    ipad[0xA]=0x36363636;
    ipad[0xB]=0x36363636;
    ipad[0xC]=0x36363636;
    ipad[0xD]=0x36363636;
    ipad[0xE]=0x36363636;
    ipad[0xF]=0x36363636;

    uint opad[16];
    opad[0x0]=0x5C5C5C5C;
    opad[0x1]=0x5C5C5C5C;
    opad[0x2]=0x5C5C5C5C;
    opad[0x3]=0x5C5C5C5C;
    opad[0x4]=0x5C5C5C5C;
    opad[0x5]=0x5C5C5C5C;
    opad[0x6]=0x5C5C5C5C;
    opad[0x7]=0x5C5C5C5C;
    opad[0x8]=0x5C5C5C5C;
    opad[0x9]=0x5C5C5C5C;
    opad[0xA]=0x5C5C5C5C;
    opad[0xB]=0x5C5C5C5C;
    opad[0xC]=0x5C5C5C5C;
    opad[0xD]=0x5C5C5C5C;
    opad[0xE]=0x5C5C5C5C;
    opad[0xF]=0x5C5C5C5C;

    #pragma unroll CONST_BYT_ACTUAL_PWLEN
    for (int m=0;m<plen && m<16;m++)
    {
        ipad[m]^=SWAP(pass[m]);
        opad[m]^=SWAP(pass[m]);
    }

    // precompute ipad
            uint stateipad[5]={0};
            stateipad[0] = 0x67452301;
            stateipad[1] = 0xefcdab89;
            stateipad[2] = 0x98badcfe;
            stateipad[3] = 0x10325476;
            stateipad[4] = 0xc3d2e1f0;

            //->sha256_update(state,W,ilenor,wposr,ipad,0x40);
            uint W[0x10]={0};
            W[0]=ipad[0];
            W[1]=ipad[1];
            W[2]=ipad[2];
            W[3]=ipad[3];
            W[4]=ipad[4];
            W[5]=ipad[5];
            W[6]=ipad[6];
            W[7]=ipad[7];
            W[8]=ipad[8];
            W[9]=ipad[9];
            W[10]=ipad[10];
            W[11]=ipad[11];
            W[12]=ipad[12];
            W[13]=ipad[13];
            W[14]=ipad[14];
            W[15]=ipad[15];
            sha1_process2(W,stateipad);

        // precompute ipad
            uint stateopad[5]={0};
            stateopad[0] = 0x67452301;
            stateopad[1] = 0xefcdab89;
            stateopad[2] = 0x98badcfe;
            stateopad[3] = 0x10325476;
            stateopad[4] = 0xc3d2e1f0;

            //->sha1_update(state,W,ilenor,wposr,ipad,0x40);
            W[0]=opad[0];
            W[1]=opad[1];
            W[2]=opad[2];
            W[3]=opad[3];
            W[4]=opad[4];
            W[5]=opad[5];
            W[6]=opad[6];
            W[7]=opad[7];
            W[8]=opad[8];
            W[9]=opad[9];
            W[10]=opad[10];
            W[11]=opad[11];
            W[12]=opad[12];
            W[13]=opad[13];
            W[14]=opad[14];
            W[15]=opad[15];
            sha1_process2(W,stateopad);

    uint counter = 1;
    uint state[5]={0};
    
    uint tkeylen=AES_CBC_256_KEY_BYTE_SIZE;
	uint cplen=0;
	while(tkeylen>0) 
    {
		if(tkeylen > 20) cplen = 20;
		else cplen=tkeylen;
        
        //hmac_sha1_init(state,W,ileno,wpos,ipad,opad,pwd);
        //->sha1_init(state,W,ileno,wpos);
        //->sha1_update(state,W,ileno,wpos,ipad,0x40);
        state[0] = stateipad[0];
        state[1] = stateipad[1];
        state[2] = stateipad[2];
        state[3] = stateipad[3];
        state[4] = stateipad[4];
        //hmac_sha1_update(state,W,ileno,wpos,ipad,opad,salt,salt_len);
        //->sha1_update(state,W,ileno,wpos,salt,salt_len);
        //hmac_sha1_update(state,W,ileno,wpos,ipad,opad,itmp,4);
        //->sha1_update(state,W,ileno,wpos,itmp,4);
        W[0]=0;
        W[1]=0;
        W[2]=0;
        W[3]=0;
        W[4]=0;
        W[5]=0;
        W[6]=0;
        W[7]=0;
        W[8]=0;
        W[9]=0;
        W[10]=0;
        W[11]=0;
        W[12]=0;
        W[13]=0;
        W[14]=0;
        #pragma unroll 16
        for (int m=0;m<slen;m++)
        {
            W[m]=SWAP(salt[m]);
        }
        W[slen]=counter;

        uint padding=0x80<<(((salt_len+4)-((salt_len+4)/4*4))*8);
        W[((mod((salt_len+4),(16*4)))/4)]|=SWAP(padding);
            // Let's add length
        W[0x0F]=(0x40+(salt_len+4))*8;

        //W[slen+1]=0x80000000;
        //W[15]=0x54*8;
        //hmac_sha1_final(state,W,ileno,ipad,opad,digtmp);
        //->sha1_finish(state,W,ileno,&opad[0x10]);
        sha1_process2(W,state);

        //sha1(opad,0x54,digtmp);
		//->sha1_init(state,W,ileno,wpos);
		//->sha1_update(state,W,ileno,wpos,opad,0x54);
		//->sha1_finish(state,W,ileno,digtmp);
        
        W[0]=state[0];
        W[1]=state[1];
        W[2]=state[2];
        W[3]=state[3];
        W[4]=state[4];
        W[5]=0x80000000;
        W[6]=0x0;
        W[7]=0x0;
        W[8]=0x0;
        W[9]=0;
        W[10]=0;
        W[11]=0;
        W[12]=0;
        W[13]=0;
        W[14]=0;
        W[15]=0x54*8;

        state[0]=stateopad[0];
        state[1]=stateopad[1];
        state[2]=stateopad[2];
        state[3]=stateopad[3];
        state[4]=stateopad[4];

        //sha256_finish(state,W,ileno,digtmp);
        sha1_process2(W,state);

        p[0]=W[0]=state[0];
        p[1]=W[1]=state[1];
        p[2]=W[2]=state[2];
        p[3]=W[3]=state[3];
        p[4]=W[4]=state[4];

        uint M[0x10];
        //very time consuming
        #pragma unroll
        for(int j = 1; j < PBKDFITER; j++) 
        {
            //hmac_sha1(pwd,digtmp,32,digtmp);
            //->sha1_init(state,W,ilenor,wposr);
            //->sha1_update(state,W,ilenor,wposr,digtmp,32);
            //->sha1_finish(state,W,ileno,&opad[0x10]);

            W[5]=0x80000000; //Padding
            W[6]=0;
            W[7]=0;
            W[8]=0;
            W[9]=0;
            W[10]=0;
            W[11]=0;
            W[12]=0;
            W[13]=0;
            W[14]=0;
            W[15]=0x54*8;
            state[0] = stateipad[0];
            state[1] = stateipad[1];
            state[2] = stateipad[2];
            state[3] = stateipad[3];
            state[4] = stateipad[4];
            sha1_process2(W,state);

            
            
            M[0]=state[0];
            M[1]=state[1];
            M[2]=state[2];
            M[3]=state[3];
            M[4]=state[4];
            M[5]=0x80000000; //Padding
            M[6]=0;
            M[7]=0;
            M[8]=0;
            M[9]=0;
            M[10]=0;
            M[11]=0;
            M[12]=0;
            M[13]=0;
            M[14]=0;
            M[15]=0x54*8;

            //->sha1_init(state,W,ilenor,wposr);
            //->sha1_update(state,W,ilenor,wposr,opad,0x60);
            state[0] = stateopad[0];
            state[1] = stateopad[1];
            state[2] = stateopad[2];
            state[3] = stateopad[3];
            state[4] = stateopad[4];

            //->sha1_finish(state,W,ilenor,digtmp);
            sha1_process2(M,state);

            W[0]=state[0];
            W[1]=state[1];
            W[2]=state[2];
            W[3]=state[3];
            W[4]=state[4];

            p[0] ^= state[0];
            p[1] ^= state[1];
            p[2] ^= state[2];
            p[3] ^= state[3];
            p[4] ^= state[4];
        }
        
        p[0]=SWAP(p[0]);
        p[1]=SWAP(p[1]);
        p[2]=SWAP(p[2]);
        p[3]=SWAP(p[3]);
        p[4]=SWAP(p[4]);
        
        tkeylen-= cplen;
        counter++;
        p+= cplen/4;
    }
    return;
}

//----------------------------------------------AES-------------------------------------------------------

__constant uchar AES_SBox[256] = 
{
   0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
   0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
   0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
   0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
   0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
   0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
   0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
   0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
   0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
   0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
   0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
   0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
   0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
   0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
   0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
   0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
};

__constant uchar Rcon[256] = {
	0x8D, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A, 
	0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A, 0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39, 
	0x72, 0xE4, 0xD3, 0xBD, 0x61, 0xC2, 0x9F, 0x25, 0x4A, 0x94, 0x33, 0x66, 0xCC, 0x83, 0x1D, 0x3A, 
	0x74, 0xE8, 0xCB, 0x8D, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36, 0x6C, 0xD8, 
	0xAB, 0x4D, 0x9A, 0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A, 0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 
	0xC5, 0x91, 0x39, 0x72, 0xE4, 0xD3, 0xBD, 0x61, 0xC2, 0x9F, 0x25, 0x4A, 0x94, 0x33, 0x66, 0xCC, 
	0x83, 0x1D, 0x3A, 0x74, 0xE8, 0xCB, 0x8D, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 
	0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A, 0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A, 0xD4, 0xB3, 
	0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39, 0x72, 0xE4, 0xD3, 0xBD, 0x61, 0xC2, 0x9F, 0x25, 0x4A, 0x94, 
	0x33, 0x66, 0xCC, 0x83, 0x1D, 0x3A, 0x74, 0xE8, 0xCB, 0x8D, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 
	0x40, 0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A, 0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 
	0x6A, 0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39, 0x72, 0xE4, 0xD3, 0xBD, 0x61, 0xC2, 0x9F, 
	0x25, 0x4A, 0x94, 0x33, 0x66, 0xCC, 0x83, 0x1D, 0x3A, 0x74, 0xE8, 0xCB, 0x8D, 0x01, 0x02, 0x04, 
	0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A, 0x2F, 0x5E, 0xBC, 0x63, 
	0xC6, 0x97, 0x35, 0x6A, 0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39, 0x72, 0xE4, 0xD3, 0xBD, 
	0x61, 0xC2, 0x9F, 0x25, 0x4A, 0x94, 0x33, 0x66, 0xCC, 0x83, 0x1D, 0x3A, 0x74, 0xE8, 0xCB, 0x8D
};

__constant uchar InvSBox[256] = {
   0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
   0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
   0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
   0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
   0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
   0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
   0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
   0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
   0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
   0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
   0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
   0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
   0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
   0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
   0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
   0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D
};

__constant uchar fieldNine[] = {
	0x00, 0x09, 0x12, 0x1b, 0x24, 0x2d, 0x36, 0x3f, 0x48, 0x41, 0x5a, 0x53, 0x6c, 0x65, 0x7e, 0x77, 
	0x90, 0x99, 0x82, 0x8b, 0xb4, 0xbd, 0xa6, 0xaf, 0xd8, 0xd1, 0xca, 0xc3, 0xfc, 0xf5, 0xee, 0xe7, 
	0x3b, 0x32, 0x29, 0x20, 0x1f, 0x16, 0x0d, 0x04, 0x73, 0x7a, 0x61, 0x68, 0x57, 0x5e, 0x45, 0x4c, 
	0xab, 0xa2, 0xb9, 0xb0, 0x8f, 0x86, 0x9d, 0x94, 0xe3, 0xea, 0xf1, 0xf8, 0xc7, 0xce, 0xd5, 0xdc, 
	0x76, 0x7f, 0x64, 0x6d, 0x52, 0x5b, 0x40, 0x49, 0x3e, 0x37, 0x2c, 0x25, 0x1a, 0x13, 0x08, 0x01, 
	0xe6, 0xef, 0xf4, 0xfd, 0xc2, 0xcb, 0xd0, 0xd9, 0xae, 0xa7, 0xbc, 0xb5, 0x8a, 0x83, 0x98, 0x91, 
	0x4d, 0x44, 0x5f, 0x56, 0x69, 0x60, 0x7b, 0x72, 0x05, 0x0c, 0x17, 0x1e, 0x21, 0x28, 0x33, 0x3a, 
	0xdd, 0xd4, 0xcf, 0xc6, 0xf9, 0xf0, 0xeb, 0xe2, 0x95, 0x9c, 0x87, 0x8e, 0xb1, 0xb8, 0xa3, 0xaa, 
	0xec, 0xe5, 0xfe, 0xf7, 0xc8, 0xc1, 0xda, 0xd3, 0xa4, 0xad, 0xb6, 0xbf, 0x80, 0x89, 0x92, 0x9b, 
	0x7c, 0x75, 0x6e, 0x67, 0x58, 0x51, 0x4a, 0x43, 0x34, 0x3d, 0x26, 0x2f, 0x10, 0x19, 0x02, 0x0b, 
	0xd7, 0xde, 0xc5, 0xcc, 0xf3, 0xfa, 0xe1, 0xe8, 0x9f, 0x96, 0x8d, 0x84, 0xbb, 0xb2, 0xa9, 0xa0, 
	0x47, 0x4e, 0x55, 0x5c, 0x63, 0x6a, 0x71, 0x78, 0x0f, 0x06, 0x1d, 0x14, 0x2b, 0x22, 0x39, 0x30, 
	0x9a, 0x93, 0x88, 0x81, 0xbe, 0xb7, 0xac, 0xa5, 0xd2, 0xdb, 0xc0, 0xc9, 0xf6, 0xff, 0xe4, 0xed, 
	0x0a, 0x03, 0x18, 0x11, 0x2e, 0x27, 0x3c, 0x35, 0x42, 0x4b, 0x50, 0x59, 0x66, 0x6f, 0x74, 0x7d, 
	0xa1, 0xa8, 0xb3, 0xba, 0x85, 0x8c, 0x97, 0x9e, 0xe9, 0xe0, 0xfb, 0xf2, 0xcd, 0xc4, 0xdf, 0xd6, 
	0x31, 0x38, 0x23, 0x2a, 0x15, 0x1c, 0x07, 0x0e, 0x79, 0x70, 0x6b, 0x62, 0x5d, 0x54, 0x4f, 0x46
};

__constant uchar fieldEleven[] = {
	0x00, 0x0b, 0x16, 0x1d, 0x2c, 0x27, 0x3a, 0x31, 0x58, 0x53, 0x4e, 0x45, 0x74, 0x7f, 0x62, 0x69, 
	0xb0, 0xbb, 0xa6, 0xad, 0x9c, 0x97, 0x8a, 0x81, 0xe8, 0xe3, 0xfe, 0xf5, 0xc4, 0xcf, 0xd2, 0xd9, 
	0x7b, 0x70, 0x6d, 0x66, 0x57, 0x5c, 0x41, 0x4a, 0x23, 0x28, 0x35, 0x3e, 0x0f, 0x04, 0x19, 0x12, 
	0xcb, 0xc0, 0xdd, 0xd6, 0xe7, 0xec, 0xf1, 0xfa, 0x93, 0x98, 0x85, 0x8e, 0xbf, 0xb4, 0xa9, 0xa2, 
	0xf6, 0xfd, 0xe0, 0xeb, 0xda, 0xd1, 0xcc, 0xc7, 0xae, 0xa5, 0xb8, 0xb3, 0x82, 0x89, 0x94, 0x9f, 
	0x46, 0x4d, 0x50, 0x5b, 0x6a, 0x61, 0x7c, 0x77, 0x1e, 0x15, 0x08, 0x03, 0x32, 0x39, 0x24, 0x2f, 
	0x8d, 0x86, 0x9b, 0x90, 0xa1, 0xaa, 0xb7, 0xbc, 0xd5, 0xde, 0xc3, 0xc8, 0xf9, 0xf2, 0xef, 0xe4, 
	0x3d, 0x36, 0x2b, 0x20, 0x11, 0x1a, 0x07, 0x0c, 0x65, 0x6e, 0x73, 0x78, 0x49, 0x42, 0x5f, 0x54, 
	0xf7, 0xfc, 0xe1, 0xea, 0xdb, 0xd0, 0xcd, 0xc6, 0xaf, 0xa4, 0xb9, 0xb2, 0x83, 0x88, 0x95, 0x9e, 
	0x47, 0x4c, 0x51, 0x5a, 0x6b, 0x60, 0x7d, 0x76, 0x1f, 0x14, 0x09, 0x02, 0x33, 0x38, 0x25, 0x2e, 
	0x8c, 0x87, 0x9a, 0x91, 0xa0, 0xab, 0xb6, 0xbd, 0xd4, 0xdf, 0xc2, 0xc9, 0xf8, 0xf3, 0xee, 0xe5, 
	0x3c, 0x37, 0x2a, 0x21, 0x10, 0x1b, 0x06, 0x0d, 0x64, 0x6f, 0x72, 0x79, 0x48, 0x43, 0x5e, 0x55, 
	0x01, 0x0a, 0x17, 0x1c, 0x2d, 0x26, 0x3b, 0x30, 0x59, 0x52, 0x4f, 0x44, 0x75, 0x7e, 0x63, 0x68, 
	0xb1, 0xba, 0xa7, 0xac, 0x9d, 0x96, 0x8b, 0x80, 0xe9, 0xe2, 0xff, 0xf4, 0xc5, 0xce, 0xd3, 0xd8, 
	0x7a, 0x71, 0x6c, 0x67, 0x56, 0x5d, 0x40, 0x4b, 0x22, 0x29, 0x34, 0x3f, 0x0e, 0x05, 0x18, 0x13, 
	0xca, 0xc1, 0xdc, 0xd7, 0xe6, 0xed, 0xf0, 0xfb, 0x92, 0x99, 0x84, 0x8f, 0xbe, 0xb5, 0xa8, 0xa3
};

__constant uchar fieldThirteen[] = {
	0x00, 0x0d, 0x1a, 0x17, 0x34, 0x39, 0x2e, 0x23, 0x68, 0x65, 0x72, 0x7f, 0x5c, 0x51, 0x46, 0x4b, 
	0xd0, 0xdd, 0xca, 0xc7, 0xe4, 0xe9, 0xfe, 0xf3, 0xb8, 0xb5, 0xa2, 0xaf, 0x8c, 0x81, 0x96, 0x9b, 
	0xbb, 0xb6, 0xa1, 0xac, 0x8f, 0x82, 0x95, 0x98, 0xd3, 0xde, 0xc9, 0xc4, 0xe7, 0xea, 0xfd, 0xf0, 
	0x6b, 0x66, 0x71, 0x7c, 0x5f, 0x52, 0x45, 0x48, 0x03, 0x0e, 0x19, 0x14, 0x37, 0x3a, 0x2d, 0x20, 
	0x6d, 0x60, 0x77, 0x7a, 0x59, 0x54, 0x43, 0x4e, 0x05, 0x08, 0x1f, 0x12, 0x31, 0x3c, 0x2b, 0x26, 
	0xbd, 0xb0, 0xa7, 0xaa, 0x89, 0x84, 0x93, 0x9e, 0xd5, 0xd8, 0xcf, 0xc2, 0xe1, 0xec, 0xfb, 0xf6, 
	0xd6, 0xdb, 0xcc, 0xc1, 0xe2, 0xef, 0xf8, 0xf5, 0xbe, 0xb3, 0xa4, 0xa9, 0x8a, 0x87, 0x90, 0x9d, 
	0x06, 0x0b, 0x1c, 0x11, 0x32, 0x3f, 0x28, 0x25, 0x6e, 0x63, 0x74, 0x79, 0x5a, 0x57, 0x40, 0x4d, 
	0xda, 0xd7, 0xc0, 0xcd, 0xee, 0xe3, 0xf4, 0xf9, 0xb2, 0xbf, 0xa8, 0xa5, 0x86, 0x8b, 0x9c, 0x91, 
	0x0a, 0x07, 0x10, 0x1d, 0x3e, 0x33, 0x24, 0x29, 0x62, 0x6f, 0x78, 0x75, 0x56, 0x5b, 0x4c, 0x41, 
	0x61, 0x6c, 0x7b, 0x76, 0x55, 0x58, 0x4f, 0x42, 0x09, 0x04, 0x13, 0x1e, 0x3d, 0x30, 0x27, 0x2a, 
	0xb1, 0xbc, 0xab, 0xa6, 0x85, 0x88, 0x9f, 0x92, 0xd9, 0xd4, 0xc3, 0xce, 0xed, 0xe0, 0xf7, 0xfa, 
	0xb7, 0xba, 0xad, 0xa0, 0x83, 0x8e, 0x99, 0x94, 0xdf, 0xd2, 0xc5, 0xc8, 0xeb, 0xe6, 0xf1, 0xfc, 
	0x67, 0x6a, 0x7d, 0x70, 0x53, 0x5e, 0x49, 0x44, 0x0f, 0x02, 0x15, 0x18, 0x3b, 0x36, 0x21, 0x2c, 
	0x0c, 0x01, 0x16, 0x1b, 0x38, 0x35, 0x22, 0x2f, 0x64, 0x69, 0x7e, 0x73, 0x50, 0x5d, 0x4a, 0x47, 
	0xdc, 0xd1, 0xc6, 0xcb, 0xe8, 0xe5, 0xf2, 0xff, 0xb4, 0xb9, 0xae, 0xa3, 0x80, 0x8d, 0x9a, 0x97
};

__constant uchar fieldFourteen[] = {
	0x00, 0x0e, 0x1c, 0x12, 0x38, 0x36, 0x24, 0x2a, 0x70, 0x7e, 0x6c, 0x62, 0x48, 0x46, 0x54, 0x5a, 
	0xe0, 0xee, 0xfc, 0xf2, 0xd8, 0xd6, 0xc4, 0xca, 0x90, 0x9e, 0x8c, 0x82, 0xa8, 0xa6, 0xb4, 0xba, 
	0xdb, 0xd5, 0xc7, 0xc9, 0xe3, 0xed, 0xff, 0xf1, 0xab, 0xa5, 0xb7, 0xb9, 0x93, 0x9d, 0x8f, 0x81, 
	0x3b, 0x35, 0x27, 0x29, 0x03, 0x0d, 0x1f, 0x11, 0x4b, 0x45, 0x57, 0x59, 0x73, 0x7d, 0x6f, 0x61, 
	0xad, 0xa3, 0xb1, 0xbf, 0x95, 0x9b, 0x89, 0x87, 0xdd, 0xd3, 0xc1, 0xcf, 0xe5, 0xeb, 0xf9, 0xf7, 
	0x4d, 0x43, 0x51, 0x5f, 0x75, 0x7b, 0x69, 0x67, 0x3d, 0x33, 0x21, 0x2f, 0x05, 0x0b, 0x19, 0x17, 
	0x76, 0x78, 0x6a, 0x64, 0x4e, 0x40, 0x52, 0x5c, 0x06, 0x08, 0x1a, 0x14, 0x3e, 0x30, 0x22, 0x2c, 
	0x96, 0x98, 0x8a, 0x84, 0xae, 0xa0, 0xb2, 0xbc, 0xe6, 0xe8, 0xfa, 0xf4, 0xde, 0xd0, 0xc2, 0xcc, 
	0x41, 0x4f, 0x5d, 0x53, 0x79, 0x77, 0x65, 0x6b, 0x31, 0x3f, 0x2d, 0x23, 0x09, 0x07, 0x15, 0x1b, 
	0xa1, 0xaf, 0xbd, 0xb3, 0x99, 0x97, 0x85, 0x8b, 0xd1, 0xdf, 0xcd, 0xc3, 0xe9, 0xe7, 0xf5, 0xfb, 
	0x9a, 0x94, 0x86, 0x88, 0xa2, 0xac, 0xbe, 0xb0, 0xea, 0xe4, 0xf6, 0xf8, 0xd2, 0xdc, 0xce, 0xc0, 
	0x7a, 0x74, 0x66, 0x68, 0x42, 0x4c, 0x5e, 0x50, 0x0a, 0x04, 0x16, 0x18, 0x32, 0x3c, 0x2e, 0x20, 
	0xec, 0xe2, 0xf0, 0xfe, 0xd4, 0xda, 0xc8, 0xc6, 0x9c, 0x92, 0x80, 0x8e, 0xa4, 0xaa, 0xb8, 0xb6, 
	0x0c, 0x02, 0x10, 0x1e, 0x34, 0x3a, 0x28, 0x26, 0x7c, 0x72, 0x60, 0x6e, 0x44, 0x4a, 0x58, 0x56, 
	0x37, 0x39, 0x2b, 0x25, 0x0f, 0x01, 0x13, 0x1d, 0x47, 0x49, 0x5b, 0x55, 0x7f, 0x71, 0x63, 0x6d, 
	0xd7, 0xd9, 0xcb, 0xc5, 0xef, 0xe1, 0xf3, 0xfd, 0xa7, 0xa9, 0xbb, 0xb5, 0x9f, 0x91, 0x83, 0x8d, 
};

void ComputeRoundKeys(uchar* roundKeys, uint rounds,uchar* key)
{

	uchar rotWord[4];

	//	The first n bytes of the expanded key are simply the encryption key.
	for (uint k = 0; k < AES_CBC_256_KEY_BYTE_SIZE; k++)
	{
		roundKeys[k] = key[k];
	}

	for (int k = 1; k < (rounds); k++)
    {
        size_t offset = AES_CBC_256_KEY_BYTE_SIZE + (k - 1) * 16; // in bytes

        if (k & 1) {
            // Calculate the rotated word
            rotWord[0] = AES_SBox[roundKeys[offset - 3]] ^ Rcon[(k + 1) >> 1];
            rotWord[1] = AES_SBox[roundKeys[offset - 2]];
            rotWord[2] = AES_SBox[roundKeys[offset - 1]];
            rotWord[3] = AES_SBox[roundKeys[offset - 4]];
        } else {
            rotWord[0] = AES_SBox[roundKeys[offset - 4]];
            rotWord[1] = AES_SBox[roundKeys[offset - 3]];
            rotWord[2] = AES_SBox[roundKeys[offset - 2]];
            rotWord[3] = AES_SBox[roundKeys[offset - 1]];
        }

        // First word
        roundKeys[offset + 0] = roundKeys[offset - 32] ^ rotWord[0];
        roundKeys[offset + 1] = roundKeys[offset - 31] ^ rotWord[1];
        roundKeys[offset + 2] = roundKeys[offset - 30] ^ rotWord[2];
        roundKeys[offset + 3] = roundKeys[offset - 29] ^ rotWord[3];

        // Second, third and forth words
        ((uint *)roundKeys)[offset/4 + 1] =
                ((uint *)roundKeys)[offset/4 + 0] ^
                ((uint *)roundKeys)[offset/4 - 7];
        ((uint *)roundKeys)[offset/4 + 2] =
                ((uint *)roundKeys)[offset/4 + 1] ^
                ((uint *)roundKeys)[offset/4 - 6];
        ((uint *)roundKeys)[offset/4 + 3] =
                ((uint *)roundKeys)[offset/4 + 2] ^
                ((uint *)roundKeys)[offset/4 - 5];
    }
}

void AddRoundKey( uchar *rkey,uchar* block, const uint round ) {
	for( uint i = 0; i < ENC_DATA_BLOCKSIZE; i++ ) block[i] ^= rkey[round*ENC_DATA_BLOCKSIZE+i];
}

void InverseShiftRows(uchar* block ) {
	uchar temp[ENC_DATA_BLOCKSIZE];
	for( uint i = 0; i < ENC_DATA_BLOCKSIZE; i++ ) temp[i] = block[i];
	
	for (uint i = 0; i < ENC_DATA_BLOCKSIZE; i++) {
		uint k = (i - (i % 4 * 4)) % ENC_DATA_BLOCKSIZE;
		block[i] = temp[k];
	}
}

void InverseSubBytes(uchar* block ) {
	for( uint i = 0; i < ENC_DATA_BLOCKSIZE; i++ ) block[i] = InvSBox[block[i]];
}

void InverseMixColumn(uchar* column,uint pos ) {
	const uchar a = fieldFourteen[column[pos+0]] ^ fieldNine[column[pos+3]] ^ fieldThirteen[column[pos+2]] ^ fieldEleven[column[pos+1]];
	const uchar b = fieldFourteen[column[pos+1]] ^ fieldNine[column[pos+0]] ^ fieldThirteen[column[pos+3]] ^ fieldEleven[column[pos+2]];
	const uchar c = fieldFourteen[column[pos+2]] ^ fieldNine[column[pos+1]] ^ fieldThirteen[column[pos+0]] ^ fieldEleven[column[pos+3]];
	const uchar d = fieldFourteen[column[pos+3]] ^ fieldNine[column[pos+2]] ^ fieldThirteen[column[pos+1]] ^ fieldEleven[column[pos+0]];
	
	column[pos+0] = a; column[pos+1] = b; column[pos+2] = c; column[pos+3] = d;
}

void InverseMixColumns(uchar* block ) {
	InverseMixColumn( block, 0 );
	InverseMixColumn( block, 4 );
	InverseMixColumn( block, 8 );
	InverseMixColumn( block, 12 );
}



//----------------------------------------------Kernel-------------------------------------------------------
__kernel void func_pbkdf2(__constant const ulong * pwstart, __constant const uint * salt,__constant const uchar* iv, __constant const uchar* oridata, __global bool * result)
{
    ulong idx = get_global_id(0);

    //calculate hex char to enum the passwords for sqlcipher db
    ulong pwidx=idx+*pwstart;
    uchar pw[CONST_BYT_ACTUAL_PWLEN]={0};
    for(ulong i=0;i<=CONST_BYT_ACTUAL_PWLEN-1;i++){
        ulong val=pwidx%16;
        if(val>=10)
        {
            pw[CONST_BYT_ACTUAL_PWLEN-1-i]=val+87;//hex char, sqlcipher db uses lower case of abcdef, a means 10, 'a'=97, offset=97-10=87
        }else
        {
            pw[CONST_BYT_ACTUAL_PWLEN-1-i]=val+48;//numbers, 0 means 0, '0'=48, offset=48-0=48
        }
        pwidx=pwidx/16;
    }

    //-------------------------PBKDF2-SHA1 Actions----------------------
    
    uint hash[AES_CBC_256_KEY_BYTE_SIZE/4]={0};//Store the pbkdf2 results
    pbkdf((uint*)pw, CONST_BYT_ACTUAL_PWLEN , salt, CONST_BYT_SALTLEN, hash);//Do pbkdf2_sha1.After that, "hash" got the correct 32byte key for AES-256-CBC.

    //-------------------------AES Actions------------------------------
    uchar* key=(uchar*)hash;//let rkey be the pointer of uchar,1/byte each place ,rkey=32byte
    uchar data[ENC_DATA_BLOCKSIZE]={0};//AES Block size(size),always 16

    #pragma unroll ENC_DATA_BLOCKSIZE
    for(uint i=0;i<ENC_DATA_BLOCKSIZE;i++)
    {
        data[i]=oridata[i];//Copy the data obtained from the host, because the functions below will modify the data as the output.
    }

    uchar rkey[256]={0};//store roundkey,actually we only use 240, 256 is to ensure not to overflow

    uint rounds=0;//rounds of roundkey
    switch (AES_CBC_256_KEY_BYTE_SIZE)
	{
	case 16: rounds = 11; break;
	case 24: rounds = 12; break;
	case 32: rounds = 15; break;
	}

    //core AES decryption
    ComputeRoundKeys(rkey,rounds,key);

	AddRoundKey( rkey, data, rounds-1 );
    
	for( uint j = 1; j < rounds-1; ++j ){
		const uint round = rounds-1 - j;
		InverseShiftRows( data );
		InverseSubBytes( data );
		AddRoundKey( rkey, data, round );
		InverseMixColumns( data );
	}
	InverseSubBytes( data );
	InverseShiftRows( data );
	AddRoundKey( rkey, data, 0 );
    
	//for sqlcipherv2 standard, verify the first 4 B at GPU side is reasonably enough, so we only need to decrypt the first block of data, and the finally decrypted text should be the data^iv(CBS Mode)
    if(((uint)(data[0] ^ iv[0])==4) && ((uint)(data[1] ^ iv[1])==0) && ((uint)(data[2] ^ iv[2])==1) && ((uint)(data[3] ^ iv[3])==1))
	{
		result[idx]=true;
	}
    
}
