/*
 *    This file was auto-generated using the ACADO Toolkit.
 *    
 *    While ACADO Toolkit is free software released under the terms of
 *    the GNU Lesser General Public License (LGPL), the generated code
 *    as such remains the property of the user who used ACADO Toolkit
 *    to generate this code. In particular, user dependent data of the code
 *    do not inherit the GNU LGPL license. On the other hand, parts of the
 *    generated code that are a direct copy of source code from the
 *    ACADO Toolkit or the software tools it is based on, remain, as derived
 *    work, automatically covered by the LGPL license.
 *    
 *    ACADO Toolkit is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *    
 */


#include "acado_common.h"




/******************************************************************************/
/*                                                                            */
/* ACADO code generation                                                      */
/*                                                                            */
/******************************************************************************/


/** Row vector of size: 30 */
real_t state[ 30 ];

int acado_modelSimulation(  )
{
int ret;

int lRun1;
ret = 0;
#pragma omp parallel for private(lRun1, state) shared(acadoWorkspace, acadoVariables)
for (lRun1 = 0; lRun1 < 50; ++lRun1)
{
state[0] = acadoVariables.x[lRun1 * 4];
state[1] = acadoVariables.x[lRun1 * 4 + 1];
state[2] = acadoVariables.x[lRun1 * 4 + 2];
state[3] = acadoVariables.x[lRun1 * 4 + 3];

state[28] = acadoVariables.u[lRun1 * 2];
state[29] = acadoVariables.u[lRun1 * 2 + 1];

ret = acado_integrate(state, 1);

acadoWorkspace.d[lRun1 * 4] = state[0] - acadoVariables.x[lRun1 * 4 + 4];
acadoWorkspace.d[lRun1 * 4 + 1] = state[1] - acadoVariables.x[lRun1 * 4 + 5];
acadoWorkspace.d[lRun1 * 4 + 2] = state[2] - acadoVariables.x[lRun1 * 4 + 6];
acadoWorkspace.d[lRun1 * 4 + 3] = state[3] - acadoVariables.x[lRun1 * 4 + 7];

acadoWorkspace.evGx[lRun1 * 16] = state[4];
acadoWorkspace.evGx[lRun1 * 16 + 1] = state[5];
acadoWorkspace.evGx[lRun1 * 16 + 2] = state[6];
acadoWorkspace.evGx[lRun1 * 16 + 3] = state[7];
acadoWorkspace.evGx[lRun1 * 16 + 4] = state[8];
acadoWorkspace.evGx[lRun1 * 16 + 5] = state[9];
acadoWorkspace.evGx[lRun1 * 16 + 6] = state[10];
acadoWorkspace.evGx[lRun1 * 16 + 7] = state[11];
acadoWorkspace.evGx[lRun1 * 16 + 8] = state[12];
acadoWorkspace.evGx[lRun1 * 16 + 9] = state[13];
acadoWorkspace.evGx[lRun1 * 16 + 10] = state[14];
acadoWorkspace.evGx[lRun1 * 16 + 11] = state[15];
acadoWorkspace.evGx[lRun1 * 16 + 12] = state[16];
acadoWorkspace.evGx[lRun1 * 16 + 13] = state[17];
acadoWorkspace.evGx[lRun1 * 16 + 14] = state[18];
acadoWorkspace.evGx[lRun1 * 16 + 15] = state[19];

acadoWorkspace.evGu[lRun1 * 8] = state[20];
acadoWorkspace.evGu[lRun1 * 8 + 1] = state[21];
acadoWorkspace.evGu[lRun1 * 8 + 2] = state[22];
acadoWorkspace.evGu[lRun1 * 8 + 3] = state[23];
acadoWorkspace.evGu[lRun1 * 8 + 4] = state[24];
acadoWorkspace.evGu[lRun1 * 8 + 5] = state[25];
acadoWorkspace.evGu[lRun1 * 8 + 6] = state[26];
acadoWorkspace.evGu[lRun1 * 8 + 7] = state[27];
}
return ret;
}

void acado_evaluateLSQ(const real_t* in, real_t* out)
{
const real_t* xd = in;

/* Compute outputs: */
out[0] = xd[0];
out[1] = xd[1];
out[2] = xd[2];
out[3] = xd[3];
}

void acado_evaluateLSQEndTerm(const real_t* in, real_t* out)
{
const real_t* xd = in;

/* Compute outputs: */
out[0] = xd[0];
out[1] = xd[1];
out[2] = xd[2];
out[3] = xd[3];
}

void acado_setObjQ1Q2( real_t* const tmpObjS, real_t* const tmpQ1, real_t* const tmpQ2 )
{
tmpQ2[0] = +tmpObjS[0];
tmpQ2[1] = +tmpObjS[1];
tmpQ2[2] = +tmpObjS[2];
tmpQ2[3] = +tmpObjS[3];
tmpQ2[4] = +tmpObjS[4];
tmpQ2[5] = +tmpObjS[5];
tmpQ2[6] = +tmpObjS[6];
tmpQ2[7] = +tmpObjS[7];
tmpQ2[8] = +tmpObjS[8];
tmpQ2[9] = +tmpObjS[9];
tmpQ2[10] = +tmpObjS[10];
tmpQ2[11] = +tmpObjS[11];
tmpQ2[12] = +tmpObjS[12];
tmpQ2[13] = +tmpObjS[13];
tmpQ2[14] = +tmpObjS[14];
tmpQ2[15] = +tmpObjS[15];
tmpQ1[0] = + tmpQ2[0];
tmpQ1[1] = + tmpQ2[1];
tmpQ1[2] = + tmpQ2[2];
tmpQ1[3] = + tmpQ2[3];
tmpQ1[4] = + tmpQ2[4];
tmpQ1[5] = + tmpQ2[5];
tmpQ1[6] = + tmpQ2[6];
tmpQ1[7] = + tmpQ2[7];
tmpQ1[8] = + tmpQ2[8];
tmpQ1[9] = + tmpQ2[9];
tmpQ1[10] = + tmpQ2[10];
tmpQ1[11] = + tmpQ2[11];
tmpQ1[12] = + tmpQ2[12];
tmpQ1[13] = + tmpQ2[13];
tmpQ1[14] = + tmpQ2[14];
tmpQ1[15] = + tmpQ2[15];
}

void acado_setObjQN1QN2( real_t* const tmpObjSEndTerm, real_t* const tmpQN1, real_t* const tmpQN2 )
{
tmpQN2[0] = +tmpObjSEndTerm[0];
tmpQN2[1] = +tmpObjSEndTerm[1];
tmpQN2[2] = +tmpObjSEndTerm[2];
tmpQN2[3] = +tmpObjSEndTerm[3];
tmpQN2[4] = +tmpObjSEndTerm[4];
tmpQN2[5] = +tmpObjSEndTerm[5];
tmpQN2[6] = +tmpObjSEndTerm[6];
tmpQN2[7] = +tmpObjSEndTerm[7];
tmpQN2[8] = +tmpObjSEndTerm[8];
tmpQN2[9] = +tmpObjSEndTerm[9];
tmpQN2[10] = +tmpObjSEndTerm[10];
tmpQN2[11] = +tmpObjSEndTerm[11];
tmpQN2[12] = +tmpObjSEndTerm[12];
tmpQN2[13] = +tmpObjSEndTerm[13];
tmpQN2[14] = +tmpObjSEndTerm[14];
tmpQN2[15] = +tmpObjSEndTerm[15];
tmpQN1[0] = + tmpQN2[0];
tmpQN1[1] = + tmpQN2[1];
tmpQN1[2] = + tmpQN2[2];
tmpQN1[3] = + tmpQN2[3];
tmpQN1[4] = + tmpQN2[4];
tmpQN1[5] = + tmpQN2[5];
tmpQN1[6] = + tmpQN2[6];
tmpQN1[7] = + tmpQN2[7];
tmpQN1[8] = + tmpQN2[8];
tmpQN1[9] = + tmpQN2[9];
tmpQN1[10] = + tmpQN2[10];
tmpQN1[11] = + tmpQN2[11];
tmpQN1[12] = + tmpQN2[12];
tmpQN1[13] = + tmpQN2[13];
tmpQN1[14] = + tmpQN2[14];
tmpQN1[15] = + tmpQN2[15];
}

void acado_evaluateObjective(  )
{
int runObj;
for (runObj = 0; runObj < 50; ++runObj)
{
acadoWorkspace.objValueIn[0] = acadoVariables.x[runObj * 4];
acadoWorkspace.objValueIn[1] = acadoVariables.x[runObj * 4 + 1];
acadoWorkspace.objValueIn[2] = acadoVariables.x[runObj * 4 + 2];
acadoWorkspace.objValueIn[3] = acadoVariables.x[runObj * 4 + 3];
acadoWorkspace.objValueIn[4] = acadoVariables.u[runObj * 2];
acadoWorkspace.objValueIn[5] = acadoVariables.u[runObj * 2 + 1];

acado_evaluateLSQ( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.Dy[runObj * 4] = acadoWorkspace.objValueOut[0];
acadoWorkspace.Dy[runObj * 4 + 1] = acadoWorkspace.objValueOut[1];
acadoWorkspace.Dy[runObj * 4 + 2] = acadoWorkspace.objValueOut[2];
acadoWorkspace.Dy[runObj * 4 + 3] = acadoWorkspace.objValueOut[3];

acado_setObjQ1Q2( &(acadoVariables.W[ runObj * 16 ]), &(acadoWorkspace.Q1[ runObj * 16 ]), &(acadoWorkspace.Q2[ runObj * 16 ]) );

}
acadoWorkspace.objValueIn[0] = acadoVariables.x[200];
acadoWorkspace.objValueIn[1] = acadoVariables.x[201];
acadoWorkspace.objValueIn[2] = acadoVariables.x[202];
acadoWorkspace.objValueIn[3] = acadoVariables.x[203];
acado_evaluateLSQEndTerm( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );

acadoWorkspace.DyN[0] = acadoWorkspace.objValueOut[0];
acadoWorkspace.DyN[1] = acadoWorkspace.objValueOut[1];
acadoWorkspace.DyN[2] = acadoWorkspace.objValueOut[2];
acadoWorkspace.DyN[3] = acadoWorkspace.objValueOut[3];

acado_setObjQN1QN2( acadoVariables.WN, acadoWorkspace.QN1, acadoWorkspace.QN2 );

}

void acado_multGxGu( real_t* const Gx1, real_t* const Gu1, real_t* const Gu2 )
{
Gu2[0] = + Gx1[0]*Gu1[0] + Gx1[1]*Gu1[2] + Gx1[2]*Gu1[4] + Gx1[3]*Gu1[6];
Gu2[1] = + Gx1[0]*Gu1[1] + Gx1[1]*Gu1[3] + Gx1[2]*Gu1[5] + Gx1[3]*Gu1[7];
Gu2[2] = + Gx1[4]*Gu1[0] + Gx1[5]*Gu1[2] + Gx1[6]*Gu1[4] + Gx1[7]*Gu1[6];
Gu2[3] = + Gx1[4]*Gu1[1] + Gx1[5]*Gu1[3] + Gx1[6]*Gu1[5] + Gx1[7]*Gu1[7];
Gu2[4] = + Gx1[8]*Gu1[0] + Gx1[9]*Gu1[2] + Gx1[10]*Gu1[4] + Gx1[11]*Gu1[6];
Gu2[5] = + Gx1[8]*Gu1[1] + Gx1[9]*Gu1[3] + Gx1[10]*Gu1[5] + Gx1[11]*Gu1[7];
Gu2[6] = + Gx1[12]*Gu1[0] + Gx1[13]*Gu1[2] + Gx1[14]*Gu1[4] + Gx1[15]*Gu1[6];
Gu2[7] = + Gx1[12]*Gu1[1] + Gx1[13]*Gu1[3] + Gx1[14]*Gu1[5] + Gx1[15]*Gu1[7];
}

void acado_moveGuE( real_t* const Gu1, real_t* const Gu2 )
{
Gu2[0] = Gu1[0];
Gu2[1] = Gu1[1];
Gu2[2] = Gu1[2];
Gu2[3] = Gu1[3];
Gu2[4] = Gu1[4];
Gu2[5] = Gu1[5];
Gu2[6] = Gu1[6];
Gu2[7] = Gu1[7];
}

void acado_multBTW1( real_t* const Gu1, real_t* const Gu2, int iRow, int iCol )
{
acadoWorkspace.H[(iRow * 200) + (iCol * 2)] = + Gu1[0]*Gu2[0] + Gu1[2]*Gu2[2] + Gu1[4]*Gu2[4] + Gu1[6]*Gu2[6];
acadoWorkspace.H[(iRow * 200) + (iCol * 2 + 1)] = + Gu1[0]*Gu2[1] + Gu1[2]*Gu2[3] + Gu1[4]*Gu2[5] + Gu1[6]*Gu2[7];
acadoWorkspace.H[(iRow * 200 + 100) + (iCol * 2)] = + Gu1[1]*Gu2[0] + Gu1[3]*Gu2[2] + Gu1[5]*Gu2[4] + Gu1[7]*Gu2[6];
acadoWorkspace.H[(iRow * 200 + 100) + (iCol * 2 + 1)] = + Gu1[1]*Gu2[1] + Gu1[3]*Gu2[3] + Gu1[5]*Gu2[5] + Gu1[7]*Gu2[7];
}

void acado_multBTW1_R1( real_t* const Gu1, real_t* const Gu2, int iRow )
{
acadoWorkspace.H[iRow * 202] = + Gu1[0]*Gu2[0] + Gu1[2]*Gu2[2] + Gu1[4]*Gu2[4] + Gu1[6]*Gu2[6];
acadoWorkspace.H[iRow * 202 + 1] = + Gu1[0]*Gu2[1] + Gu1[2]*Gu2[3] + Gu1[4]*Gu2[5] + Gu1[6]*Gu2[7];
acadoWorkspace.H[iRow * 202 + 100] = + Gu1[1]*Gu2[0] + Gu1[3]*Gu2[2] + Gu1[5]*Gu2[4] + Gu1[7]*Gu2[6];
acadoWorkspace.H[iRow * 202 + 101] = + Gu1[1]*Gu2[1] + Gu1[3]*Gu2[3] + Gu1[5]*Gu2[5] + Gu1[7]*Gu2[7];
}

void acado_multGxTGu( real_t* const Gx1, real_t* const Gu1, real_t* const Gu2 )
{
Gu2[0] = + Gx1[0]*Gu1[0] + Gx1[4]*Gu1[2] + Gx1[8]*Gu1[4] + Gx1[12]*Gu1[6];
Gu2[1] = + Gx1[0]*Gu1[1] + Gx1[4]*Gu1[3] + Gx1[8]*Gu1[5] + Gx1[12]*Gu1[7];
Gu2[2] = + Gx1[1]*Gu1[0] + Gx1[5]*Gu1[2] + Gx1[9]*Gu1[4] + Gx1[13]*Gu1[6];
Gu2[3] = + Gx1[1]*Gu1[1] + Gx1[5]*Gu1[3] + Gx1[9]*Gu1[5] + Gx1[13]*Gu1[7];
Gu2[4] = + Gx1[2]*Gu1[0] + Gx1[6]*Gu1[2] + Gx1[10]*Gu1[4] + Gx1[14]*Gu1[6];
Gu2[5] = + Gx1[2]*Gu1[1] + Gx1[6]*Gu1[3] + Gx1[10]*Gu1[5] + Gx1[14]*Gu1[7];
Gu2[6] = + Gx1[3]*Gu1[0] + Gx1[7]*Gu1[2] + Gx1[11]*Gu1[4] + Gx1[15]*Gu1[6];
Gu2[7] = + Gx1[3]*Gu1[1] + Gx1[7]*Gu1[3] + Gx1[11]*Gu1[5] + Gx1[15]*Gu1[7];
}

void acado_multQEW2( real_t* const Q11, real_t* const Gu1, real_t* const Gu2, real_t* const Gu3 )
{
Gu3[0] = + Q11[0]*Gu1[0] + Q11[1]*Gu1[2] + Q11[2]*Gu1[4] + Q11[3]*Gu1[6] + Gu2[0];
Gu3[1] = + Q11[0]*Gu1[1] + Q11[1]*Gu1[3] + Q11[2]*Gu1[5] + Q11[3]*Gu1[7] + Gu2[1];
Gu3[2] = + Q11[4]*Gu1[0] + Q11[5]*Gu1[2] + Q11[6]*Gu1[4] + Q11[7]*Gu1[6] + Gu2[2];
Gu3[3] = + Q11[4]*Gu1[1] + Q11[5]*Gu1[3] + Q11[6]*Gu1[5] + Q11[7]*Gu1[7] + Gu2[3];
Gu3[4] = + Q11[8]*Gu1[0] + Q11[9]*Gu1[2] + Q11[10]*Gu1[4] + Q11[11]*Gu1[6] + Gu2[4];
Gu3[5] = + Q11[8]*Gu1[1] + Q11[9]*Gu1[3] + Q11[10]*Gu1[5] + Q11[11]*Gu1[7] + Gu2[5];
Gu3[6] = + Q11[12]*Gu1[0] + Q11[13]*Gu1[2] + Q11[14]*Gu1[4] + Q11[15]*Gu1[6] + Gu2[6];
Gu3[7] = + Q11[12]*Gu1[1] + Q11[13]*Gu1[3] + Q11[14]*Gu1[5] + Q11[15]*Gu1[7] + Gu2[7];
}

void acado_macATw1QDy( real_t* const Gx1, real_t* const w11, real_t* const w12, real_t* const w13 )
{
w13[0] = + Gx1[0]*w11[0] + Gx1[4]*w11[1] + Gx1[8]*w11[2] + Gx1[12]*w11[3] + w12[0];
w13[1] = + Gx1[1]*w11[0] + Gx1[5]*w11[1] + Gx1[9]*w11[2] + Gx1[13]*w11[3] + w12[1];
w13[2] = + Gx1[2]*w11[0] + Gx1[6]*w11[1] + Gx1[10]*w11[2] + Gx1[14]*w11[3] + w12[2];
w13[3] = + Gx1[3]*w11[0] + Gx1[7]*w11[1] + Gx1[11]*w11[2] + Gx1[15]*w11[3] + w12[3];
}

void acado_macBTw1( real_t* const Gu1, real_t* const w11, real_t* const U1 )
{
U1[0] += + Gu1[0]*w11[0] + Gu1[2]*w11[1] + Gu1[4]*w11[2] + Gu1[6]*w11[3];
U1[1] += + Gu1[1]*w11[0] + Gu1[3]*w11[1] + Gu1[5]*w11[2] + Gu1[7]*w11[3];
}

void acado_macQSbarW2( real_t* const Q11, real_t* const w11, real_t* const w12, real_t* const w13 )
{
w13[0] = + Q11[0]*w11[0] + Q11[1]*w11[1] + Q11[2]*w11[2] + Q11[3]*w11[3] + w12[0];
w13[1] = + Q11[4]*w11[0] + Q11[5]*w11[1] + Q11[6]*w11[2] + Q11[7]*w11[3] + w12[1];
w13[2] = + Q11[8]*w11[0] + Q11[9]*w11[1] + Q11[10]*w11[2] + Q11[11]*w11[3] + w12[2];
w13[3] = + Q11[12]*w11[0] + Q11[13]*w11[1] + Q11[14]*w11[2] + Q11[15]*w11[3] + w12[3];
}

void acado_macASbar( real_t* const Gx1, real_t* const w11, real_t* const w12 )
{
w12[0] += + Gx1[0]*w11[0] + Gx1[1]*w11[1] + Gx1[2]*w11[2] + Gx1[3]*w11[3];
w12[1] += + Gx1[4]*w11[0] + Gx1[5]*w11[1] + Gx1[6]*w11[2] + Gx1[7]*w11[3];
w12[2] += + Gx1[8]*w11[0] + Gx1[9]*w11[1] + Gx1[10]*w11[2] + Gx1[11]*w11[3];
w12[3] += + Gx1[12]*w11[0] + Gx1[13]*w11[1] + Gx1[14]*w11[2] + Gx1[15]*w11[3];
}

void acado_expansionStep( real_t* const Gx1, real_t* const Gu1, real_t* const U1, real_t* const w11, real_t* const w12 )
{
w12[0] += + Gx1[0]*w11[0] + Gx1[1]*w11[1] + Gx1[2]*w11[2] + Gx1[3]*w11[3];
w12[1] += + Gx1[4]*w11[0] + Gx1[5]*w11[1] + Gx1[6]*w11[2] + Gx1[7]*w11[3];
w12[2] += + Gx1[8]*w11[0] + Gx1[9]*w11[1] + Gx1[10]*w11[2] + Gx1[11]*w11[3];
w12[3] += + Gx1[12]*w11[0] + Gx1[13]*w11[1] + Gx1[14]*w11[2] + Gx1[15]*w11[3];
w12[0] += + Gu1[0]*U1[0] + Gu1[1]*U1[1];
w12[1] += + Gu1[2]*U1[0] + Gu1[3]*U1[1];
w12[2] += + Gu1[4]*U1[0] + Gu1[5]*U1[1];
w12[3] += + Gu1[6]*U1[0] + Gu1[7]*U1[1];
}

void acado_copyHTH( int iRow, int iCol )
{
acadoWorkspace.H[(iRow * 200) + (iCol * 2)] = acadoWorkspace.H[(iCol * 200) + (iRow * 2)];
acadoWorkspace.H[(iRow * 200) + (iCol * 2 + 1)] = acadoWorkspace.H[(iCol * 200 + 100) + (iRow * 2)];
acadoWorkspace.H[(iRow * 200 + 100) + (iCol * 2)] = acadoWorkspace.H[(iCol * 200) + (iRow * 2 + 1)];
acadoWorkspace.H[(iRow * 200 + 100) + (iCol * 2 + 1)] = acadoWorkspace.H[(iCol * 200 + 100) + (iRow * 2 + 1)];
}

void acado_multRDy( real_t* const Dy1, real_t* const RDy1 )
{
RDy1[0] = 0.0;
;
RDy1[1] = 0.0;
;
}

void acado_multQDy( real_t* const Q2, real_t* const Dy1, real_t* const QDy1 )
{
QDy1[0] = + Q2[0]*Dy1[0] + Q2[1]*Dy1[1] + Q2[2]*Dy1[2] + Q2[3]*Dy1[3];
QDy1[1] = + Q2[4]*Dy1[0] + Q2[5]*Dy1[1] + Q2[6]*Dy1[2] + Q2[7]*Dy1[3];
QDy1[2] = + Q2[8]*Dy1[0] + Q2[9]*Dy1[1] + Q2[10]*Dy1[2] + Q2[11]*Dy1[3];
QDy1[3] = + Q2[12]*Dy1[0] + Q2[13]*Dy1[1] + Q2[14]*Dy1[2] + Q2[15]*Dy1[3];
}

void acado_condensePrep(  )
{
int lRun1;
int lRun2;
int lRun3;
for (lRun2 = 0; lRun2 < 50; ++lRun2)
{
lRun3 = ((lRun2) * (lRun2 * -1 + 101)) / (2);
acado_moveGuE( &(acadoWorkspace.evGu[ lRun2 * 8 ]), &(acadoWorkspace.E[ lRun3 * 8 ]) );
for (lRun1 = 1; lRun1 < lRun2 * -1 + 50; ++lRun1)
{
acado_multGxGu( &(acadoWorkspace.evGx[ ((((lRun2) + (lRun1)) * (4)) * (4)) + (0) ]), &(acadoWorkspace.E[ (((((lRun3) + (lRun1)) - (1)) * (4)) * (2)) + (0) ]), &(acadoWorkspace.E[ ((((lRun3) + (lRun1)) * (4)) * (2)) + (0) ]) );
}

acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ ((((((lRun3) - (lRun2)) + (50)) - (1)) * (4)) * (2)) + (0) ]), acadoWorkspace.W1 );
for (lRun1 = 49; lRun2 < lRun1; --lRun1)
{
acado_multBTW1( &(acadoWorkspace.evGu[ lRun1 * 8 ]), acadoWorkspace.W1, lRun1, lRun2 );
acado_multGxTGu( &(acadoWorkspace.evGx[ lRun1 * 16 ]), acadoWorkspace.W1, acadoWorkspace.W2 );
acado_multQEW2( &(acadoWorkspace.Q1[ lRun1 * 16 ]), &(acadoWorkspace.E[ ((((((lRun3) + (lRun1)) - (lRun2)) - (1)) * (4)) * (2)) + (0) ]), acadoWorkspace.W2, acadoWorkspace.W1 );
}
acado_multBTW1_R1( &(acadoWorkspace.evGu[ lRun2 * 8 ]), acadoWorkspace.W1, lRun2 );
}

for (lRun1 = 0; lRun1 < 50; ++lRun1)
{
for (lRun2 = 0; lRun2 < lRun1; ++lRun2)
{
acado_copyHTH( lRun2, lRun1 );
}
}

for (lRun1 = 0; lRun1 < 200; ++lRun1)
acadoWorkspace.sbar[lRun1 + 4] = acadoWorkspace.d[lRun1];


}

void acado_condenseFdb(  )
{
int lRun1;
acadoWorkspace.Dx0[0] = acadoVariables.x0[0] - acadoVariables.x[0];
acadoWorkspace.Dx0[1] = acadoVariables.x0[1] - acadoVariables.x[1];
acadoWorkspace.Dx0[2] = acadoVariables.x0[2] - acadoVariables.x[2];
acadoWorkspace.Dx0[3] = acadoVariables.x0[3] - acadoVariables.x[3];
for (lRun1 = 0; lRun1 < 200; ++lRun1)
acadoWorkspace.Dy[lRun1] -= acadoVariables.y[lRun1];

acadoWorkspace.DyN[0] -= acadoVariables.yN[0];
acadoWorkspace.DyN[1] -= acadoVariables.yN[1];
acadoWorkspace.DyN[2] -= acadoVariables.yN[2];
acadoWorkspace.DyN[3] -= acadoVariables.yN[3];

acado_multRDy( acadoWorkspace.Dy, acadoWorkspace.g );
acado_multRDy( &(acadoWorkspace.Dy[ 4 ]), &(acadoWorkspace.g[ 2 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 8 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 12 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 16 ]), &(acadoWorkspace.g[ 8 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 20 ]), &(acadoWorkspace.g[ 10 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 24 ]), &(acadoWorkspace.g[ 12 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 28 ]), &(acadoWorkspace.g[ 14 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 32 ]), &(acadoWorkspace.g[ 16 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 36 ]), &(acadoWorkspace.g[ 18 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 40 ]), &(acadoWorkspace.g[ 20 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 44 ]), &(acadoWorkspace.g[ 22 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 48 ]), &(acadoWorkspace.g[ 24 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 52 ]), &(acadoWorkspace.g[ 26 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 56 ]), &(acadoWorkspace.g[ 28 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 60 ]), &(acadoWorkspace.g[ 30 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 64 ]), &(acadoWorkspace.g[ 32 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 68 ]), &(acadoWorkspace.g[ 34 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 72 ]), &(acadoWorkspace.g[ 36 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 76 ]), &(acadoWorkspace.g[ 38 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 80 ]), &(acadoWorkspace.g[ 40 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 84 ]), &(acadoWorkspace.g[ 42 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 88 ]), &(acadoWorkspace.g[ 44 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 92 ]), &(acadoWorkspace.g[ 46 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 96 ]), &(acadoWorkspace.g[ 48 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 100 ]), &(acadoWorkspace.g[ 50 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 104 ]), &(acadoWorkspace.g[ 52 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 108 ]), &(acadoWorkspace.g[ 54 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 112 ]), &(acadoWorkspace.g[ 56 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 116 ]), &(acadoWorkspace.g[ 58 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 120 ]), &(acadoWorkspace.g[ 60 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 124 ]), &(acadoWorkspace.g[ 62 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 128 ]), &(acadoWorkspace.g[ 64 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 132 ]), &(acadoWorkspace.g[ 66 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 136 ]), &(acadoWorkspace.g[ 68 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 140 ]), &(acadoWorkspace.g[ 70 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 144 ]), &(acadoWorkspace.g[ 72 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 148 ]), &(acadoWorkspace.g[ 74 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 152 ]), &(acadoWorkspace.g[ 76 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 156 ]), &(acadoWorkspace.g[ 78 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 160 ]), &(acadoWorkspace.g[ 80 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 164 ]), &(acadoWorkspace.g[ 82 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 168 ]), &(acadoWorkspace.g[ 84 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 172 ]), &(acadoWorkspace.g[ 86 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 176 ]), &(acadoWorkspace.g[ 88 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 180 ]), &(acadoWorkspace.g[ 90 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 184 ]), &(acadoWorkspace.g[ 92 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 188 ]), &(acadoWorkspace.g[ 94 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 192 ]), &(acadoWorkspace.g[ 96 ]) );
acado_multRDy( &(acadoWorkspace.Dy[ 196 ]), &(acadoWorkspace.g[ 98 ]) );

acado_multQDy( acadoWorkspace.Q2, acadoWorkspace.Dy, acadoWorkspace.QDy );
acado_multQDy( &(acadoWorkspace.Q2[ 16 ]), &(acadoWorkspace.Dy[ 4 ]), &(acadoWorkspace.QDy[ 4 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 32 ]), &(acadoWorkspace.Dy[ 8 ]), &(acadoWorkspace.QDy[ 8 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 48 ]), &(acadoWorkspace.Dy[ 12 ]), &(acadoWorkspace.QDy[ 12 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 64 ]), &(acadoWorkspace.Dy[ 16 ]), &(acadoWorkspace.QDy[ 16 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 80 ]), &(acadoWorkspace.Dy[ 20 ]), &(acadoWorkspace.QDy[ 20 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 96 ]), &(acadoWorkspace.Dy[ 24 ]), &(acadoWorkspace.QDy[ 24 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 112 ]), &(acadoWorkspace.Dy[ 28 ]), &(acadoWorkspace.QDy[ 28 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 128 ]), &(acadoWorkspace.Dy[ 32 ]), &(acadoWorkspace.QDy[ 32 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 144 ]), &(acadoWorkspace.Dy[ 36 ]), &(acadoWorkspace.QDy[ 36 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 160 ]), &(acadoWorkspace.Dy[ 40 ]), &(acadoWorkspace.QDy[ 40 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 176 ]), &(acadoWorkspace.Dy[ 44 ]), &(acadoWorkspace.QDy[ 44 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 192 ]), &(acadoWorkspace.Dy[ 48 ]), &(acadoWorkspace.QDy[ 48 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 208 ]), &(acadoWorkspace.Dy[ 52 ]), &(acadoWorkspace.QDy[ 52 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 224 ]), &(acadoWorkspace.Dy[ 56 ]), &(acadoWorkspace.QDy[ 56 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 240 ]), &(acadoWorkspace.Dy[ 60 ]), &(acadoWorkspace.QDy[ 60 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 256 ]), &(acadoWorkspace.Dy[ 64 ]), &(acadoWorkspace.QDy[ 64 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 272 ]), &(acadoWorkspace.Dy[ 68 ]), &(acadoWorkspace.QDy[ 68 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 288 ]), &(acadoWorkspace.Dy[ 72 ]), &(acadoWorkspace.QDy[ 72 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 304 ]), &(acadoWorkspace.Dy[ 76 ]), &(acadoWorkspace.QDy[ 76 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 320 ]), &(acadoWorkspace.Dy[ 80 ]), &(acadoWorkspace.QDy[ 80 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 336 ]), &(acadoWorkspace.Dy[ 84 ]), &(acadoWorkspace.QDy[ 84 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 352 ]), &(acadoWorkspace.Dy[ 88 ]), &(acadoWorkspace.QDy[ 88 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 368 ]), &(acadoWorkspace.Dy[ 92 ]), &(acadoWorkspace.QDy[ 92 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 384 ]), &(acadoWorkspace.Dy[ 96 ]), &(acadoWorkspace.QDy[ 96 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 400 ]), &(acadoWorkspace.Dy[ 100 ]), &(acadoWorkspace.QDy[ 100 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 416 ]), &(acadoWorkspace.Dy[ 104 ]), &(acadoWorkspace.QDy[ 104 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 432 ]), &(acadoWorkspace.Dy[ 108 ]), &(acadoWorkspace.QDy[ 108 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 448 ]), &(acadoWorkspace.Dy[ 112 ]), &(acadoWorkspace.QDy[ 112 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 464 ]), &(acadoWorkspace.Dy[ 116 ]), &(acadoWorkspace.QDy[ 116 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 480 ]), &(acadoWorkspace.Dy[ 120 ]), &(acadoWorkspace.QDy[ 120 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 496 ]), &(acadoWorkspace.Dy[ 124 ]), &(acadoWorkspace.QDy[ 124 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 512 ]), &(acadoWorkspace.Dy[ 128 ]), &(acadoWorkspace.QDy[ 128 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 528 ]), &(acadoWorkspace.Dy[ 132 ]), &(acadoWorkspace.QDy[ 132 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 544 ]), &(acadoWorkspace.Dy[ 136 ]), &(acadoWorkspace.QDy[ 136 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 560 ]), &(acadoWorkspace.Dy[ 140 ]), &(acadoWorkspace.QDy[ 140 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 576 ]), &(acadoWorkspace.Dy[ 144 ]), &(acadoWorkspace.QDy[ 144 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 592 ]), &(acadoWorkspace.Dy[ 148 ]), &(acadoWorkspace.QDy[ 148 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 608 ]), &(acadoWorkspace.Dy[ 152 ]), &(acadoWorkspace.QDy[ 152 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 624 ]), &(acadoWorkspace.Dy[ 156 ]), &(acadoWorkspace.QDy[ 156 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 640 ]), &(acadoWorkspace.Dy[ 160 ]), &(acadoWorkspace.QDy[ 160 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 656 ]), &(acadoWorkspace.Dy[ 164 ]), &(acadoWorkspace.QDy[ 164 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 672 ]), &(acadoWorkspace.Dy[ 168 ]), &(acadoWorkspace.QDy[ 168 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 688 ]), &(acadoWorkspace.Dy[ 172 ]), &(acadoWorkspace.QDy[ 172 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 704 ]), &(acadoWorkspace.Dy[ 176 ]), &(acadoWorkspace.QDy[ 176 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 720 ]), &(acadoWorkspace.Dy[ 180 ]), &(acadoWorkspace.QDy[ 180 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 736 ]), &(acadoWorkspace.Dy[ 184 ]), &(acadoWorkspace.QDy[ 184 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 752 ]), &(acadoWorkspace.Dy[ 188 ]), &(acadoWorkspace.QDy[ 188 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 768 ]), &(acadoWorkspace.Dy[ 192 ]), &(acadoWorkspace.QDy[ 192 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 784 ]), &(acadoWorkspace.Dy[ 196 ]), &(acadoWorkspace.QDy[ 196 ]) );

acadoWorkspace.QDy[200] = + acadoWorkspace.QN2[0]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[1]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[2]*acadoWorkspace.DyN[2] + acadoWorkspace.QN2[3]*acadoWorkspace.DyN[3];
acadoWorkspace.QDy[201] = + acadoWorkspace.QN2[4]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[5]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[6]*acadoWorkspace.DyN[2] + acadoWorkspace.QN2[7]*acadoWorkspace.DyN[3];
acadoWorkspace.QDy[202] = + acadoWorkspace.QN2[8]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[9]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[10]*acadoWorkspace.DyN[2] + acadoWorkspace.QN2[11]*acadoWorkspace.DyN[3];
acadoWorkspace.QDy[203] = + acadoWorkspace.QN2[12]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[13]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[14]*acadoWorkspace.DyN[2] + acadoWorkspace.QN2[15]*acadoWorkspace.DyN[3];

acadoWorkspace.sbar[0] = acadoWorkspace.Dx0[0];
acadoWorkspace.sbar[1] = acadoWorkspace.Dx0[1];
acadoWorkspace.sbar[2] = acadoWorkspace.Dx0[2];
acadoWorkspace.sbar[3] = acadoWorkspace.Dx0[3];
acado_macASbar( acadoWorkspace.evGx, acadoWorkspace.sbar, &(acadoWorkspace.sbar[ 4 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 16 ]), &(acadoWorkspace.sbar[ 4 ]), &(acadoWorkspace.sbar[ 8 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 32 ]), &(acadoWorkspace.sbar[ 8 ]), &(acadoWorkspace.sbar[ 12 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 48 ]), &(acadoWorkspace.sbar[ 12 ]), &(acadoWorkspace.sbar[ 16 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.sbar[ 16 ]), &(acadoWorkspace.sbar[ 20 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 80 ]), &(acadoWorkspace.sbar[ 20 ]), &(acadoWorkspace.sbar[ 24 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 96 ]), &(acadoWorkspace.sbar[ 24 ]), &(acadoWorkspace.sbar[ 28 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.sbar[ 28 ]), &(acadoWorkspace.sbar[ 32 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.sbar[ 32 ]), &(acadoWorkspace.sbar[ 36 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.sbar[ 36 ]), &(acadoWorkspace.sbar[ 40 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.sbar[ 40 ]), &(acadoWorkspace.sbar[ 44 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.sbar[ 44 ]), &(acadoWorkspace.sbar[ 48 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.sbar[ 48 ]), &(acadoWorkspace.sbar[ 52 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.sbar[ 52 ]), &(acadoWorkspace.sbar[ 56 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.sbar[ 56 ]), &(acadoWorkspace.sbar[ 60 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.sbar[ 60 ]), &(acadoWorkspace.sbar[ 64 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.sbar[ 64 ]), &(acadoWorkspace.sbar[ 68 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.sbar[ 68 ]), &(acadoWorkspace.sbar[ 72 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.sbar[ 72 ]), &(acadoWorkspace.sbar[ 76 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.sbar[ 76 ]), &(acadoWorkspace.sbar[ 80 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 320 ]), &(acadoWorkspace.sbar[ 80 ]), &(acadoWorkspace.sbar[ 84 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 336 ]), &(acadoWorkspace.sbar[ 84 ]), &(acadoWorkspace.sbar[ 88 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 352 ]), &(acadoWorkspace.sbar[ 88 ]), &(acadoWorkspace.sbar[ 92 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 368 ]), &(acadoWorkspace.sbar[ 92 ]), &(acadoWorkspace.sbar[ 96 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 384 ]), &(acadoWorkspace.sbar[ 96 ]), &(acadoWorkspace.sbar[ 100 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.sbar[ 100 ]), &(acadoWorkspace.sbar[ 104 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 416 ]), &(acadoWorkspace.sbar[ 104 ]), &(acadoWorkspace.sbar[ 108 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 432 ]), &(acadoWorkspace.sbar[ 108 ]), &(acadoWorkspace.sbar[ 112 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 448 ]), &(acadoWorkspace.sbar[ 112 ]), &(acadoWorkspace.sbar[ 116 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 464 ]), &(acadoWorkspace.sbar[ 116 ]), &(acadoWorkspace.sbar[ 120 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 480 ]), &(acadoWorkspace.sbar[ 120 ]), &(acadoWorkspace.sbar[ 124 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 496 ]), &(acadoWorkspace.sbar[ 124 ]), &(acadoWorkspace.sbar[ 128 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 512 ]), &(acadoWorkspace.sbar[ 128 ]), &(acadoWorkspace.sbar[ 132 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 528 ]), &(acadoWorkspace.sbar[ 132 ]), &(acadoWorkspace.sbar[ 136 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 544 ]), &(acadoWorkspace.sbar[ 136 ]), &(acadoWorkspace.sbar[ 140 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 560 ]), &(acadoWorkspace.sbar[ 140 ]), &(acadoWorkspace.sbar[ 144 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.sbar[ 144 ]), &(acadoWorkspace.sbar[ 148 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 592 ]), &(acadoWorkspace.sbar[ 148 ]), &(acadoWorkspace.sbar[ 152 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 608 ]), &(acadoWorkspace.sbar[ 152 ]), &(acadoWorkspace.sbar[ 156 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 624 ]), &(acadoWorkspace.sbar[ 156 ]), &(acadoWorkspace.sbar[ 160 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 640 ]), &(acadoWorkspace.sbar[ 160 ]), &(acadoWorkspace.sbar[ 164 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 656 ]), &(acadoWorkspace.sbar[ 164 ]), &(acadoWorkspace.sbar[ 168 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 672 ]), &(acadoWorkspace.sbar[ 168 ]), &(acadoWorkspace.sbar[ 172 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 688 ]), &(acadoWorkspace.sbar[ 172 ]), &(acadoWorkspace.sbar[ 176 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 704 ]), &(acadoWorkspace.sbar[ 176 ]), &(acadoWorkspace.sbar[ 180 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 720 ]), &(acadoWorkspace.sbar[ 180 ]), &(acadoWorkspace.sbar[ 184 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 736 ]), &(acadoWorkspace.sbar[ 184 ]), &(acadoWorkspace.sbar[ 188 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 752 ]), &(acadoWorkspace.sbar[ 188 ]), &(acadoWorkspace.sbar[ 192 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 768 ]), &(acadoWorkspace.sbar[ 192 ]), &(acadoWorkspace.sbar[ 196 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 784 ]), &(acadoWorkspace.sbar[ 196 ]), &(acadoWorkspace.sbar[ 200 ]) );

acadoWorkspace.w1[0] = + acadoWorkspace.QN1[0]*acadoWorkspace.sbar[200] + acadoWorkspace.QN1[1]*acadoWorkspace.sbar[201] + acadoWorkspace.QN1[2]*acadoWorkspace.sbar[202] + acadoWorkspace.QN1[3]*acadoWorkspace.sbar[203] + acadoWorkspace.QDy[200];
acadoWorkspace.w1[1] = + acadoWorkspace.QN1[4]*acadoWorkspace.sbar[200] + acadoWorkspace.QN1[5]*acadoWorkspace.sbar[201] + acadoWorkspace.QN1[6]*acadoWorkspace.sbar[202] + acadoWorkspace.QN1[7]*acadoWorkspace.sbar[203] + acadoWorkspace.QDy[201];
acadoWorkspace.w1[2] = + acadoWorkspace.QN1[8]*acadoWorkspace.sbar[200] + acadoWorkspace.QN1[9]*acadoWorkspace.sbar[201] + acadoWorkspace.QN1[10]*acadoWorkspace.sbar[202] + acadoWorkspace.QN1[11]*acadoWorkspace.sbar[203] + acadoWorkspace.QDy[202];
acadoWorkspace.w1[3] = + acadoWorkspace.QN1[12]*acadoWorkspace.sbar[200] + acadoWorkspace.QN1[13]*acadoWorkspace.sbar[201] + acadoWorkspace.QN1[14]*acadoWorkspace.sbar[202] + acadoWorkspace.QN1[15]*acadoWorkspace.sbar[203] + acadoWorkspace.QDy[203];
acado_macBTw1( &(acadoWorkspace.evGu[ 392 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 98 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 784 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 196 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 784 ]), &(acadoWorkspace.sbar[ 196 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 384 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 96 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 768 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 192 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 768 ]), &(acadoWorkspace.sbar[ 192 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 376 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 94 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 752 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 188 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 752 ]), &(acadoWorkspace.sbar[ 188 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 368 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 92 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 736 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 184 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 736 ]), &(acadoWorkspace.sbar[ 184 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 360 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 90 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 720 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 180 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 720 ]), &(acadoWorkspace.sbar[ 180 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 352 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 88 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 704 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 176 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 704 ]), &(acadoWorkspace.sbar[ 176 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 344 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 86 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 688 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 172 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 688 ]), &(acadoWorkspace.sbar[ 172 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 336 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 84 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 672 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 168 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 672 ]), &(acadoWorkspace.sbar[ 168 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 328 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 82 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 656 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 164 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 656 ]), &(acadoWorkspace.sbar[ 164 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 320 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 80 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 640 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 160 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 640 ]), &(acadoWorkspace.sbar[ 160 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 312 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 78 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 624 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 156 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 624 ]), &(acadoWorkspace.sbar[ 156 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 304 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 76 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 608 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 152 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 608 ]), &(acadoWorkspace.sbar[ 152 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 296 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 74 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 592 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 148 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 592 ]), &(acadoWorkspace.sbar[ 148 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 288 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 72 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 576 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 144 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 576 ]), &(acadoWorkspace.sbar[ 144 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 280 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 70 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 560 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 140 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 560 ]), &(acadoWorkspace.sbar[ 140 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 272 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 68 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 544 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 136 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 544 ]), &(acadoWorkspace.sbar[ 136 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 264 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 66 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 528 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 132 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 528 ]), &(acadoWorkspace.sbar[ 132 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 256 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 64 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 512 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 128 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 512 ]), &(acadoWorkspace.sbar[ 128 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 248 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 62 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 496 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 124 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 496 ]), &(acadoWorkspace.sbar[ 124 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 240 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 60 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 480 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 120 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 480 ]), &(acadoWorkspace.sbar[ 120 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 232 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 58 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 464 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 116 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 464 ]), &(acadoWorkspace.sbar[ 116 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 224 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 56 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 448 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 112 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 448 ]), &(acadoWorkspace.sbar[ 112 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 216 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 54 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 432 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 108 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 432 ]), &(acadoWorkspace.sbar[ 108 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 208 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 52 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 416 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 104 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 416 ]), &(acadoWorkspace.sbar[ 104 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 200 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 50 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 400 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 100 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 400 ]), &(acadoWorkspace.sbar[ 100 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 192 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 48 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 384 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 96 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 384 ]), &(acadoWorkspace.sbar[ 96 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 184 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 46 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 368 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 92 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 368 ]), &(acadoWorkspace.sbar[ 92 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 176 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 44 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 352 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 88 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 352 ]), &(acadoWorkspace.sbar[ 88 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 168 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 42 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 336 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 84 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 336 ]), &(acadoWorkspace.sbar[ 84 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 160 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 40 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 320 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 80 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 320 ]), &(acadoWorkspace.sbar[ 80 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 152 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 38 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 304 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 76 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 304 ]), &(acadoWorkspace.sbar[ 76 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 144 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 36 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 288 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 72 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 288 ]), &(acadoWorkspace.sbar[ 72 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 136 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 34 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 272 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 68 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 272 ]), &(acadoWorkspace.sbar[ 68 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 128 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 32 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 256 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 64 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 256 ]), &(acadoWorkspace.sbar[ 64 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 120 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 30 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 240 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 60 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 240 ]), &(acadoWorkspace.sbar[ 60 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 112 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 28 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 224 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 56 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 224 ]), &(acadoWorkspace.sbar[ 56 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 104 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 26 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 208 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 52 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 208 ]), &(acadoWorkspace.sbar[ 52 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 96 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 24 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 192 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 48 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 192 ]), &(acadoWorkspace.sbar[ 48 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 88 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 22 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 176 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 44 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 176 ]), &(acadoWorkspace.sbar[ 44 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 80 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 20 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 160 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 40 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 160 ]), &(acadoWorkspace.sbar[ 40 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 72 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 18 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 144 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 36 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 144 ]), &(acadoWorkspace.sbar[ 36 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 64 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 16 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 128 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 32 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 128 ]), &(acadoWorkspace.sbar[ 32 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 56 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 14 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 112 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 28 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 112 ]), &(acadoWorkspace.sbar[ 28 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 48 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 12 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 96 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 24 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 96 ]), &(acadoWorkspace.sbar[ 24 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 40 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 10 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 80 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 20 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 80 ]), &(acadoWorkspace.sbar[ 20 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 32 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 8 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 64 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 16 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 64 ]), &(acadoWorkspace.sbar[ 16 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 24 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 6 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 48 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 12 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 48 ]), &(acadoWorkspace.sbar[ 12 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 16 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 4 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 32 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 8 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 32 ]), &(acadoWorkspace.sbar[ 8 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 8 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 2 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 16 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 4 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 16 ]), &(acadoWorkspace.sbar[ 4 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( acadoWorkspace.evGu, acadoWorkspace.w1, acadoWorkspace.g );

acadoWorkspace.lb[0] = acadoVariables.lbValues[0] - acadoVariables.u[0];
acadoWorkspace.lb[1] = acadoVariables.lbValues[1] - acadoVariables.u[1];
acadoWorkspace.lb[2] = acadoVariables.lbValues[2] - acadoVariables.u[2];
acadoWorkspace.lb[3] = acadoVariables.lbValues[3] - acadoVariables.u[3];
acadoWorkspace.lb[4] = acadoVariables.lbValues[4] - acadoVariables.u[4];
acadoWorkspace.lb[5] = acadoVariables.lbValues[5] - acadoVariables.u[5];
acadoWorkspace.lb[6] = acadoVariables.lbValues[6] - acadoVariables.u[6];
acadoWorkspace.lb[7] = acadoVariables.lbValues[7] - acadoVariables.u[7];
acadoWorkspace.lb[8] = acadoVariables.lbValues[8] - acadoVariables.u[8];
acadoWorkspace.lb[9] = acadoVariables.lbValues[9] - acadoVariables.u[9];
acadoWorkspace.lb[10] = acadoVariables.lbValues[10] - acadoVariables.u[10];
acadoWorkspace.lb[11] = acadoVariables.lbValues[11] - acadoVariables.u[11];
acadoWorkspace.lb[12] = acadoVariables.lbValues[12] - acadoVariables.u[12];
acadoWorkspace.lb[13] = acadoVariables.lbValues[13] - acadoVariables.u[13];
acadoWorkspace.lb[14] = acadoVariables.lbValues[14] - acadoVariables.u[14];
acadoWorkspace.lb[15] = acadoVariables.lbValues[15] - acadoVariables.u[15];
acadoWorkspace.lb[16] = acadoVariables.lbValues[16] - acadoVariables.u[16];
acadoWorkspace.lb[17] = acadoVariables.lbValues[17] - acadoVariables.u[17];
acadoWorkspace.lb[18] = acadoVariables.lbValues[18] - acadoVariables.u[18];
acadoWorkspace.lb[19] = acadoVariables.lbValues[19] - acadoVariables.u[19];
acadoWorkspace.lb[20] = acadoVariables.lbValues[20] - acadoVariables.u[20];
acadoWorkspace.lb[21] = acadoVariables.lbValues[21] - acadoVariables.u[21];
acadoWorkspace.lb[22] = acadoVariables.lbValues[22] - acadoVariables.u[22];
acadoWorkspace.lb[23] = acadoVariables.lbValues[23] - acadoVariables.u[23];
acadoWorkspace.lb[24] = acadoVariables.lbValues[24] - acadoVariables.u[24];
acadoWorkspace.lb[25] = acadoVariables.lbValues[25] - acadoVariables.u[25];
acadoWorkspace.lb[26] = acadoVariables.lbValues[26] - acadoVariables.u[26];
acadoWorkspace.lb[27] = acadoVariables.lbValues[27] - acadoVariables.u[27];
acadoWorkspace.lb[28] = acadoVariables.lbValues[28] - acadoVariables.u[28];
acadoWorkspace.lb[29] = acadoVariables.lbValues[29] - acadoVariables.u[29];
acadoWorkspace.lb[30] = acadoVariables.lbValues[30] - acadoVariables.u[30];
acadoWorkspace.lb[31] = acadoVariables.lbValues[31] - acadoVariables.u[31];
acadoWorkspace.lb[32] = acadoVariables.lbValues[32] - acadoVariables.u[32];
acadoWorkspace.lb[33] = acadoVariables.lbValues[33] - acadoVariables.u[33];
acadoWorkspace.lb[34] = acadoVariables.lbValues[34] - acadoVariables.u[34];
acadoWorkspace.lb[35] = acadoVariables.lbValues[35] - acadoVariables.u[35];
acadoWorkspace.lb[36] = acadoVariables.lbValues[36] - acadoVariables.u[36];
acadoWorkspace.lb[37] = acadoVariables.lbValues[37] - acadoVariables.u[37];
acadoWorkspace.lb[38] = acadoVariables.lbValues[38] - acadoVariables.u[38];
acadoWorkspace.lb[39] = acadoVariables.lbValues[39] - acadoVariables.u[39];
acadoWorkspace.lb[40] = acadoVariables.lbValues[40] - acadoVariables.u[40];
acadoWorkspace.lb[41] = acadoVariables.lbValues[41] - acadoVariables.u[41];
acadoWorkspace.lb[42] = acadoVariables.lbValues[42] - acadoVariables.u[42];
acadoWorkspace.lb[43] = acadoVariables.lbValues[43] - acadoVariables.u[43];
acadoWorkspace.lb[44] = acadoVariables.lbValues[44] - acadoVariables.u[44];
acadoWorkspace.lb[45] = acadoVariables.lbValues[45] - acadoVariables.u[45];
acadoWorkspace.lb[46] = acadoVariables.lbValues[46] - acadoVariables.u[46];
acadoWorkspace.lb[47] = acadoVariables.lbValues[47] - acadoVariables.u[47];
acadoWorkspace.lb[48] = acadoVariables.lbValues[48] - acadoVariables.u[48];
acadoWorkspace.lb[49] = acadoVariables.lbValues[49] - acadoVariables.u[49];
acadoWorkspace.lb[50] = acadoVariables.lbValues[50] - acadoVariables.u[50];
acadoWorkspace.lb[51] = acadoVariables.lbValues[51] - acadoVariables.u[51];
acadoWorkspace.lb[52] = acadoVariables.lbValues[52] - acadoVariables.u[52];
acadoWorkspace.lb[53] = acadoVariables.lbValues[53] - acadoVariables.u[53];
acadoWorkspace.lb[54] = acadoVariables.lbValues[54] - acadoVariables.u[54];
acadoWorkspace.lb[55] = acadoVariables.lbValues[55] - acadoVariables.u[55];
acadoWorkspace.lb[56] = acadoVariables.lbValues[56] - acadoVariables.u[56];
acadoWorkspace.lb[57] = acadoVariables.lbValues[57] - acadoVariables.u[57];
acadoWorkspace.lb[58] = acadoVariables.lbValues[58] - acadoVariables.u[58];
acadoWorkspace.lb[59] = acadoVariables.lbValues[59] - acadoVariables.u[59];
acadoWorkspace.lb[60] = acadoVariables.lbValues[60] - acadoVariables.u[60];
acadoWorkspace.lb[61] = acadoVariables.lbValues[61] - acadoVariables.u[61];
acadoWorkspace.lb[62] = acadoVariables.lbValues[62] - acadoVariables.u[62];
acadoWorkspace.lb[63] = acadoVariables.lbValues[63] - acadoVariables.u[63];
acadoWorkspace.lb[64] = acadoVariables.lbValues[64] - acadoVariables.u[64];
acadoWorkspace.lb[65] = acadoVariables.lbValues[65] - acadoVariables.u[65];
acadoWorkspace.lb[66] = acadoVariables.lbValues[66] - acadoVariables.u[66];
acadoWorkspace.lb[67] = acadoVariables.lbValues[67] - acadoVariables.u[67];
acadoWorkspace.lb[68] = acadoVariables.lbValues[68] - acadoVariables.u[68];
acadoWorkspace.lb[69] = acadoVariables.lbValues[69] - acadoVariables.u[69];
acadoWorkspace.lb[70] = acadoVariables.lbValues[70] - acadoVariables.u[70];
acadoWorkspace.lb[71] = acadoVariables.lbValues[71] - acadoVariables.u[71];
acadoWorkspace.lb[72] = acadoVariables.lbValues[72] - acadoVariables.u[72];
acadoWorkspace.lb[73] = acadoVariables.lbValues[73] - acadoVariables.u[73];
acadoWorkspace.lb[74] = acadoVariables.lbValues[74] - acadoVariables.u[74];
acadoWorkspace.lb[75] = acadoVariables.lbValues[75] - acadoVariables.u[75];
acadoWorkspace.lb[76] = acadoVariables.lbValues[76] - acadoVariables.u[76];
acadoWorkspace.lb[77] = acadoVariables.lbValues[77] - acadoVariables.u[77];
acadoWorkspace.lb[78] = acadoVariables.lbValues[78] - acadoVariables.u[78];
acadoWorkspace.lb[79] = acadoVariables.lbValues[79] - acadoVariables.u[79];
acadoWorkspace.lb[80] = acadoVariables.lbValues[80] - acadoVariables.u[80];
acadoWorkspace.lb[81] = acadoVariables.lbValues[81] - acadoVariables.u[81];
acadoWorkspace.lb[82] = acadoVariables.lbValues[82] - acadoVariables.u[82];
acadoWorkspace.lb[83] = acadoVariables.lbValues[83] - acadoVariables.u[83];
acadoWorkspace.lb[84] = acadoVariables.lbValues[84] - acadoVariables.u[84];
acadoWorkspace.lb[85] = acadoVariables.lbValues[85] - acadoVariables.u[85];
acadoWorkspace.lb[86] = acadoVariables.lbValues[86] - acadoVariables.u[86];
acadoWorkspace.lb[87] = acadoVariables.lbValues[87] - acadoVariables.u[87];
acadoWorkspace.lb[88] = acadoVariables.lbValues[88] - acadoVariables.u[88];
acadoWorkspace.lb[89] = acadoVariables.lbValues[89] - acadoVariables.u[89];
acadoWorkspace.lb[90] = acadoVariables.lbValues[90] - acadoVariables.u[90];
acadoWorkspace.lb[91] = acadoVariables.lbValues[91] - acadoVariables.u[91];
acadoWorkspace.lb[92] = acadoVariables.lbValues[92] - acadoVariables.u[92];
acadoWorkspace.lb[93] = acadoVariables.lbValues[93] - acadoVariables.u[93];
acadoWorkspace.lb[94] = acadoVariables.lbValues[94] - acadoVariables.u[94];
acadoWorkspace.lb[95] = acadoVariables.lbValues[95] - acadoVariables.u[95];
acadoWorkspace.lb[96] = acadoVariables.lbValues[96] - acadoVariables.u[96];
acadoWorkspace.lb[97] = acadoVariables.lbValues[97] - acadoVariables.u[97];
acadoWorkspace.lb[98] = acadoVariables.lbValues[98] - acadoVariables.u[98];
acadoWorkspace.lb[99] = acadoVariables.lbValues[99] - acadoVariables.u[99];
acadoWorkspace.ub[0] = acadoVariables.ubValues[0] - acadoVariables.u[0];
acadoWorkspace.ub[1] = acadoVariables.ubValues[1] - acadoVariables.u[1];
acadoWorkspace.ub[2] = acadoVariables.ubValues[2] - acadoVariables.u[2];
acadoWorkspace.ub[3] = acadoVariables.ubValues[3] - acadoVariables.u[3];
acadoWorkspace.ub[4] = acadoVariables.ubValues[4] - acadoVariables.u[4];
acadoWorkspace.ub[5] = acadoVariables.ubValues[5] - acadoVariables.u[5];
acadoWorkspace.ub[6] = acadoVariables.ubValues[6] - acadoVariables.u[6];
acadoWorkspace.ub[7] = acadoVariables.ubValues[7] - acadoVariables.u[7];
acadoWorkspace.ub[8] = acadoVariables.ubValues[8] - acadoVariables.u[8];
acadoWorkspace.ub[9] = acadoVariables.ubValues[9] - acadoVariables.u[9];
acadoWorkspace.ub[10] = acadoVariables.ubValues[10] - acadoVariables.u[10];
acadoWorkspace.ub[11] = acadoVariables.ubValues[11] - acadoVariables.u[11];
acadoWorkspace.ub[12] = acadoVariables.ubValues[12] - acadoVariables.u[12];
acadoWorkspace.ub[13] = acadoVariables.ubValues[13] - acadoVariables.u[13];
acadoWorkspace.ub[14] = acadoVariables.ubValues[14] - acadoVariables.u[14];
acadoWorkspace.ub[15] = acadoVariables.ubValues[15] - acadoVariables.u[15];
acadoWorkspace.ub[16] = acadoVariables.ubValues[16] - acadoVariables.u[16];
acadoWorkspace.ub[17] = acadoVariables.ubValues[17] - acadoVariables.u[17];
acadoWorkspace.ub[18] = acadoVariables.ubValues[18] - acadoVariables.u[18];
acadoWorkspace.ub[19] = acadoVariables.ubValues[19] - acadoVariables.u[19];
acadoWorkspace.ub[20] = acadoVariables.ubValues[20] - acadoVariables.u[20];
acadoWorkspace.ub[21] = acadoVariables.ubValues[21] - acadoVariables.u[21];
acadoWorkspace.ub[22] = acadoVariables.ubValues[22] - acadoVariables.u[22];
acadoWorkspace.ub[23] = acadoVariables.ubValues[23] - acadoVariables.u[23];
acadoWorkspace.ub[24] = acadoVariables.ubValues[24] - acadoVariables.u[24];
acadoWorkspace.ub[25] = acadoVariables.ubValues[25] - acadoVariables.u[25];
acadoWorkspace.ub[26] = acadoVariables.ubValues[26] - acadoVariables.u[26];
acadoWorkspace.ub[27] = acadoVariables.ubValues[27] - acadoVariables.u[27];
acadoWorkspace.ub[28] = acadoVariables.ubValues[28] - acadoVariables.u[28];
acadoWorkspace.ub[29] = acadoVariables.ubValues[29] - acadoVariables.u[29];
acadoWorkspace.ub[30] = acadoVariables.ubValues[30] - acadoVariables.u[30];
acadoWorkspace.ub[31] = acadoVariables.ubValues[31] - acadoVariables.u[31];
acadoWorkspace.ub[32] = acadoVariables.ubValues[32] - acadoVariables.u[32];
acadoWorkspace.ub[33] = acadoVariables.ubValues[33] - acadoVariables.u[33];
acadoWorkspace.ub[34] = acadoVariables.ubValues[34] - acadoVariables.u[34];
acadoWorkspace.ub[35] = acadoVariables.ubValues[35] - acadoVariables.u[35];
acadoWorkspace.ub[36] = acadoVariables.ubValues[36] - acadoVariables.u[36];
acadoWorkspace.ub[37] = acadoVariables.ubValues[37] - acadoVariables.u[37];
acadoWorkspace.ub[38] = acadoVariables.ubValues[38] - acadoVariables.u[38];
acadoWorkspace.ub[39] = acadoVariables.ubValues[39] - acadoVariables.u[39];
acadoWorkspace.ub[40] = acadoVariables.ubValues[40] - acadoVariables.u[40];
acadoWorkspace.ub[41] = acadoVariables.ubValues[41] - acadoVariables.u[41];
acadoWorkspace.ub[42] = acadoVariables.ubValues[42] - acadoVariables.u[42];
acadoWorkspace.ub[43] = acadoVariables.ubValues[43] - acadoVariables.u[43];
acadoWorkspace.ub[44] = acadoVariables.ubValues[44] - acadoVariables.u[44];
acadoWorkspace.ub[45] = acadoVariables.ubValues[45] - acadoVariables.u[45];
acadoWorkspace.ub[46] = acadoVariables.ubValues[46] - acadoVariables.u[46];
acadoWorkspace.ub[47] = acadoVariables.ubValues[47] - acadoVariables.u[47];
acadoWorkspace.ub[48] = acadoVariables.ubValues[48] - acadoVariables.u[48];
acadoWorkspace.ub[49] = acadoVariables.ubValues[49] - acadoVariables.u[49];
acadoWorkspace.ub[50] = acadoVariables.ubValues[50] - acadoVariables.u[50];
acadoWorkspace.ub[51] = acadoVariables.ubValues[51] - acadoVariables.u[51];
acadoWorkspace.ub[52] = acadoVariables.ubValues[52] - acadoVariables.u[52];
acadoWorkspace.ub[53] = acadoVariables.ubValues[53] - acadoVariables.u[53];
acadoWorkspace.ub[54] = acadoVariables.ubValues[54] - acadoVariables.u[54];
acadoWorkspace.ub[55] = acadoVariables.ubValues[55] - acadoVariables.u[55];
acadoWorkspace.ub[56] = acadoVariables.ubValues[56] - acadoVariables.u[56];
acadoWorkspace.ub[57] = acadoVariables.ubValues[57] - acadoVariables.u[57];
acadoWorkspace.ub[58] = acadoVariables.ubValues[58] - acadoVariables.u[58];
acadoWorkspace.ub[59] = acadoVariables.ubValues[59] - acadoVariables.u[59];
acadoWorkspace.ub[60] = acadoVariables.ubValues[60] - acadoVariables.u[60];
acadoWorkspace.ub[61] = acadoVariables.ubValues[61] - acadoVariables.u[61];
acadoWorkspace.ub[62] = acadoVariables.ubValues[62] - acadoVariables.u[62];
acadoWorkspace.ub[63] = acadoVariables.ubValues[63] - acadoVariables.u[63];
acadoWorkspace.ub[64] = acadoVariables.ubValues[64] - acadoVariables.u[64];
acadoWorkspace.ub[65] = acadoVariables.ubValues[65] - acadoVariables.u[65];
acadoWorkspace.ub[66] = acadoVariables.ubValues[66] - acadoVariables.u[66];
acadoWorkspace.ub[67] = acadoVariables.ubValues[67] - acadoVariables.u[67];
acadoWorkspace.ub[68] = acadoVariables.ubValues[68] - acadoVariables.u[68];
acadoWorkspace.ub[69] = acadoVariables.ubValues[69] - acadoVariables.u[69];
acadoWorkspace.ub[70] = acadoVariables.ubValues[70] - acadoVariables.u[70];
acadoWorkspace.ub[71] = acadoVariables.ubValues[71] - acadoVariables.u[71];
acadoWorkspace.ub[72] = acadoVariables.ubValues[72] - acadoVariables.u[72];
acadoWorkspace.ub[73] = acadoVariables.ubValues[73] - acadoVariables.u[73];
acadoWorkspace.ub[74] = acadoVariables.ubValues[74] - acadoVariables.u[74];
acadoWorkspace.ub[75] = acadoVariables.ubValues[75] - acadoVariables.u[75];
acadoWorkspace.ub[76] = acadoVariables.ubValues[76] - acadoVariables.u[76];
acadoWorkspace.ub[77] = acadoVariables.ubValues[77] - acadoVariables.u[77];
acadoWorkspace.ub[78] = acadoVariables.ubValues[78] - acadoVariables.u[78];
acadoWorkspace.ub[79] = acadoVariables.ubValues[79] - acadoVariables.u[79];
acadoWorkspace.ub[80] = acadoVariables.ubValues[80] - acadoVariables.u[80];
acadoWorkspace.ub[81] = acadoVariables.ubValues[81] - acadoVariables.u[81];
acadoWorkspace.ub[82] = acadoVariables.ubValues[82] - acadoVariables.u[82];
acadoWorkspace.ub[83] = acadoVariables.ubValues[83] - acadoVariables.u[83];
acadoWorkspace.ub[84] = acadoVariables.ubValues[84] - acadoVariables.u[84];
acadoWorkspace.ub[85] = acadoVariables.ubValues[85] - acadoVariables.u[85];
acadoWorkspace.ub[86] = acadoVariables.ubValues[86] - acadoVariables.u[86];
acadoWorkspace.ub[87] = acadoVariables.ubValues[87] - acadoVariables.u[87];
acadoWorkspace.ub[88] = acadoVariables.ubValues[88] - acadoVariables.u[88];
acadoWorkspace.ub[89] = acadoVariables.ubValues[89] - acadoVariables.u[89];
acadoWorkspace.ub[90] = acadoVariables.ubValues[90] - acadoVariables.u[90];
acadoWorkspace.ub[91] = acadoVariables.ubValues[91] - acadoVariables.u[91];
acadoWorkspace.ub[92] = acadoVariables.ubValues[92] - acadoVariables.u[92];
acadoWorkspace.ub[93] = acadoVariables.ubValues[93] - acadoVariables.u[93];
acadoWorkspace.ub[94] = acadoVariables.ubValues[94] - acadoVariables.u[94];
acadoWorkspace.ub[95] = acadoVariables.ubValues[95] - acadoVariables.u[95];
acadoWorkspace.ub[96] = acadoVariables.ubValues[96] - acadoVariables.u[96];
acadoWorkspace.ub[97] = acadoVariables.ubValues[97] - acadoVariables.u[97];
acadoWorkspace.ub[98] = acadoVariables.ubValues[98] - acadoVariables.u[98];
acadoWorkspace.ub[99] = acadoVariables.ubValues[99] - acadoVariables.u[99];

}

void acado_expand(  )
{
int lRun1;
acadoVariables.u[0] += acadoWorkspace.x[0];
acadoVariables.u[1] += acadoWorkspace.x[1];
acadoVariables.u[2] += acadoWorkspace.x[2];
acadoVariables.u[3] += acadoWorkspace.x[3];
acadoVariables.u[4] += acadoWorkspace.x[4];
acadoVariables.u[5] += acadoWorkspace.x[5];
acadoVariables.u[6] += acadoWorkspace.x[6];
acadoVariables.u[7] += acadoWorkspace.x[7];
acadoVariables.u[8] += acadoWorkspace.x[8];
acadoVariables.u[9] += acadoWorkspace.x[9];
acadoVariables.u[10] += acadoWorkspace.x[10];
acadoVariables.u[11] += acadoWorkspace.x[11];
acadoVariables.u[12] += acadoWorkspace.x[12];
acadoVariables.u[13] += acadoWorkspace.x[13];
acadoVariables.u[14] += acadoWorkspace.x[14];
acadoVariables.u[15] += acadoWorkspace.x[15];
acadoVariables.u[16] += acadoWorkspace.x[16];
acadoVariables.u[17] += acadoWorkspace.x[17];
acadoVariables.u[18] += acadoWorkspace.x[18];
acadoVariables.u[19] += acadoWorkspace.x[19];
acadoVariables.u[20] += acadoWorkspace.x[20];
acadoVariables.u[21] += acadoWorkspace.x[21];
acadoVariables.u[22] += acadoWorkspace.x[22];
acadoVariables.u[23] += acadoWorkspace.x[23];
acadoVariables.u[24] += acadoWorkspace.x[24];
acadoVariables.u[25] += acadoWorkspace.x[25];
acadoVariables.u[26] += acadoWorkspace.x[26];
acadoVariables.u[27] += acadoWorkspace.x[27];
acadoVariables.u[28] += acadoWorkspace.x[28];
acadoVariables.u[29] += acadoWorkspace.x[29];
acadoVariables.u[30] += acadoWorkspace.x[30];
acadoVariables.u[31] += acadoWorkspace.x[31];
acadoVariables.u[32] += acadoWorkspace.x[32];
acadoVariables.u[33] += acadoWorkspace.x[33];
acadoVariables.u[34] += acadoWorkspace.x[34];
acadoVariables.u[35] += acadoWorkspace.x[35];
acadoVariables.u[36] += acadoWorkspace.x[36];
acadoVariables.u[37] += acadoWorkspace.x[37];
acadoVariables.u[38] += acadoWorkspace.x[38];
acadoVariables.u[39] += acadoWorkspace.x[39];
acadoVariables.u[40] += acadoWorkspace.x[40];
acadoVariables.u[41] += acadoWorkspace.x[41];
acadoVariables.u[42] += acadoWorkspace.x[42];
acadoVariables.u[43] += acadoWorkspace.x[43];
acadoVariables.u[44] += acadoWorkspace.x[44];
acadoVariables.u[45] += acadoWorkspace.x[45];
acadoVariables.u[46] += acadoWorkspace.x[46];
acadoVariables.u[47] += acadoWorkspace.x[47];
acadoVariables.u[48] += acadoWorkspace.x[48];
acadoVariables.u[49] += acadoWorkspace.x[49];
acadoVariables.u[50] += acadoWorkspace.x[50];
acadoVariables.u[51] += acadoWorkspace.x[51];
acadoVariables.u[52] += acadoWorkspace.x[52];
acadoVariables.u[53] += acadoWorkspace.x[53];
acadoVariables.u[54] += acadoWorkspace.x[54];
acadoVariables.u[55] += acadoWorkspace.x[55];
acadoVariables.u[56] += acadoWorkspace.x[56];
acadoVariables.u[57] += acadoWorkspace.x[57];
acadoVariables.u[58] += acadoWorkspace.x[58];
acadoVariables.u[59] += acadoWorkspace.x[59];
acadoVariables.u[60] += acadoWorkspace.x[60];
acadoVariables.u[61] += acadoWorkspace.x[61];
acadoVariables.u[62] += acadoWorkspace.x[62];
acadoVariables.u[63] += acadoWorkspace.x[63];
acadoVariables.u[64] += acadoWorkspace.x[64];
acadoVariables.u[65] += acadoWorkspace.x[65];
acadoVariables.u[66] += acadoWorkspace.x[66];
acadoVariables.u[67] += acadoWorkspace.x[67];
acadoVariables.u[68] += acadoWorkspace.x[68];
acadoVariables.u[69] += acadoWorkspace.x[69];
acadoVariables.u[70] += acadoWorkspace.x[70];
acadoVariables.u[71] += acadoWorkspace.x[71];
acadoVariables.u[72] += acadoWorkspace.x[72];
acadoVariables.u[73] += acadoWorkspace.x[73];
acadoVariables.u[74] += acadoWorkspace.x[74];
acadoVariables.u[75] += acadoWorkspace.x[75];
acadoVariables.u[76] += acadoWorkspace.x[76];
acadoVariables.u[77] += acadoWorkspace.x[77];
acadoVariables.u[78] += acadoWorkspace.x[78];
acadoVariables.u[79] += acadoWorkspace.x[79];
acadoVariables.u[80] += acadoWorkspace.x[80];
acadoVariables.u[81] += acadoWorkspace.x[81];
acadoVariables.u[82] += acadoWorkspace.x[82];
acadoVariables.u[83] += acadoWorkspace.x[83];
acadoVariables.u[84] += acadoWorkspace.x[84];
acadoVariables.u[85] += acadoWorkspace.x[85];
acadoVariables.u[86] += acadoWorkspace.x[86];
acadoVariables.u[87] += acadoWorkspace.x[87];
acadoVariables.u[88] += acadoWorkspace.x[88];
acadoVariables.u[89] += acadoWorkspace.x[89];
acadoVariables.u[90] += acadoWorkspace.x[90];
acadoVariables.u[91] += acadoWorkspace.x[91];
acadoVariables.u[92] += acadoWorkspace.x[92];
acadoVariables.u[93] += acadoWorkspace.x[93];
acadoVariables.u[94] += acadoWorkspace.x[94];
acadoVariables.u[95] += acadoWorkspace.x[95];
acadoVariables.u[96] += acadoWorkspace.x[96];
acadoVariables.u[97] += acadoWorkspace.x[97];
acadoVariables.u[98] += acadoWorkspace.x[98];
acadoVariables.u[99] += acadoWorkspace.x[99];
acadoWorkspace.sbar[0] = acadoWorkspace.Dx0[0];
acadoWorkspace.sbar[1] = acadoWorkspace.Dx0[1];
acadoWorkspace.sbar[2] = acadoWorkspace.Dx0[2];
acadoWorkspace.sbar[3] = acadoWorkspace.Dx0[3];
for (lRun1 = 0; lRun1 < 200; ++lRun1)
acadoWorkspace.sbar[lRun1 + 4] = acadoWorkspace.d[lRun1];

acado_expansionStep( acadoWorkspace.evGx, acadoWorkspace.evGu, acadoWorkspace.x, acadoWorkspace.sbar, &(acadoWorkspace.sbar[ 4 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 16 ]), &(acadoWorkspace.evGu[ 8 ]), &(acadoWorkspace.x[ 2 ]), &(acadoWorkspace.sbar[ 4 ]), &(acadoWorkspace.sbar[ 8 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 32 ]), &(acadoWorkspace.evGu[ 16 ]), &(acadoWorkspace.x[ 4 ]), &(acadoWorkspace.sbar[ 8 ]), &(acadoWorkspace.sbar[ 12 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 48 ]), &(acadoWorkspace.evGu[ 24 ]), &(acadoWorkspace.x[ 6 ]), &(acadoWorkspace.sbar[ 12 ]), &(acadoWorkspace.sbar[ 16 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.evGu[ 32 ]), &(acadoWorkspace.x[ 8 ]), &(acadoWorkspace.sbar[ 16 ]), &(acadoWorkspace.sbar[ 20 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 80 ]), &(acadoWorkspace.evGu[ 40 ]), &(acadoWorkspace.x[ 10 ]), &(acadoWorkspace.sbar[ 20 ]), &(acadoWorkspace.sbar[ 24 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 96 ]), &(acadoWorkspace.evGu[ 48 ]), &(acadoWorkspace.x[ 12 ]), &(acadoWorkspace.sbar[ 24 ]), &(acadoWorkspace.sbar[ 28 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 112 ]), &(acadoWorkspace.evGu[ 56 ]), &(acadoWorkspace.x[ 14 ]), &(acadoWorkspace.sbar[ 28 ]), &(acadoWorkspace.sbar[ 32 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 128 ]), &(acadoWorkspace.evGu[ 64 ]), &(acadoWorkspace.x[ 16 ]), &(acadoWorkspace.sbar[ 32 ]), &(acadoWorkspace.sbar[ 36 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 144 ]), &(acadoWorkspace.evGu[ 72 ]), &(acadoWorkspace.x[ 18 ]), &(acadoWorkspace.sbar[ 36 ]), &(acadoWorkspace.sbar[ 40 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 160 ]), &(acadoWorkspace.evGu[ 80 ]), &(acadoWorkspace.x[ 20 ]), &(acadoWorkspace.sbar[ 40 ]), &(acadoWorkspace.sbar[ 44 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 176 ]), &(acadoWorkspace.evGu[ 88 ]), &(acadoWorkspace.x[ 22 ]), &(acadoWorkspace.sbar[ 44 ]), &(acadoWorkspace.sbar[ 48 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 192 ]), &(acadoWorkspace.evGu[ 96 ]), &(acadoWorkspace.x[ 24 ]), &(acadoWorkspace.sbar[ 48 ]), &(acadoWorkspace.sbar[ 52 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 208 ]), &(acadoWorkspace.evGu[ 104 ]), &(acadoWorkspace.x[ 26 ]), &(acadoWorkspace.sbar[ 52 ]), &(acadoWorkspace.sbar[ 56 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 224 ]), &(acadoWorkspace.evGu[ 112 ]), &(acadoWorkspace.x[ 28 ]), &(acadoWorkspace.sbar[ 56 ]), &(acadoWorkspace.sbar[ 60 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 240 ]), &(acadoWorkspace.evGu[ 120 ]), &(acadoWorkspace.x[ 30 ]), &(acadoWorkspace.sbar[ 60 ]), &(acadoWorkspace.sbar[ 64 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 256 ]), &(acadoWorkspace.evGu[ 128 ]), &(acadoWorkspace.x[ 32 ]), &(acadoWorkspace.sbar[ 64 ]), &(acadoWorkspace.sbar[ 68 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 272 ]), &(acadoWorkspace.evGu[ 136 ]), &(acadoWorkspace.x[ 34 ]), &(acadoWorkspace.sbar[ 68 ]), &(acadoWorkspace.sbar[ 72 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 288 ]), &(acadoWorkspace.evGu[ 144 ]), &(acadoWorkspace.x[ 36 ]), &(acadoWorkspace.sbar[ 72 ]), &(acadoWorkspace.sbar[ 76 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 304 ]), &(acadoWorkspace.evGu[ 152 ]), &(acadoWorkspace.x[ 38 ]), &(acadoWorkspace.sbar[ 76 ]), &(acadoWorkspace.sbar[ 80 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 320 ]), &(acadoWorkspace.evGu[ 160 ]), &(acadoWorkspace.x[ 40 ]), &(acadoWorkspace.sbar[ 80 ]), &(acadoWorkspace.sbar[ 84 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 336 ]), &(acadoWorkspace.evGu[ 168 ]), &(acadoWorkspace.x[ 42 ]), &(acadoWorkspace.sbar[ 84 ]), &(acadoWorkspace.sbar[ 88 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 352 ]), &(acadoWorkspace.evGu[ 176 ]), &(acadoWorkspace.x[ 44 ]), &(acadoWorkspace.sbar[ 88 ]), &(acadoWorkspace.sbar[ 92 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 368 ]), &(acadoWorkspace.evGu[ 184 ]), &(acadoWorkspace.x[ 46 ]), &(acadoWorkspace.sbar[ 92 ]), &(acadoWorkspace.sbar[ 96 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 384 ]), &(acadoWorkspace.evGu[ 192 ]), &(acadoWorkspace.x[ 48 ]), &(acadoWorkspace.sbar[ 96 ]), &(acadoWorkspace.sbar[ 100 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 400 ]), &(acadoWorkspace.evGu[ 200 ]), &(acadoWorkspace.x[ 50 ]), &(acadoWorkspace.sbar[ 100 ]), &(acadoWorkspace.sbar[ 104 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 416 ]), &(acadoWorkspace.evGu[ 208 ]), &(acadoWorkspace.x[ 52 ]), &(acadoWorkspace.sbar[ 104 ]), &(acadoWorkspace.sbar[ 108 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 432 ]), &(acadoWorkspace.evGu[ 216 ]), &(acadoWorkspace.x[ 54 ]), &(acadoWorkspace.sbar[ 108 ]), &(acadoWorkspace.sbar[ 112 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 448 ]), &(acadoWorkspace.evGu[ 224 ]), &(acadoWorkspace.x[ 56 ]), &(acadoWorkspace.sbar[ 112 ]), &(acadoWorkspace.sbar[ 116 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 464 ]), &(acadoWorkspace.evGu[ 232 ]), &(acadoWorkspace.x[ 58 ]), &(acadoWorkspace.sbar[ 116 ]), &(acadoWorkspace.sbar[ 120 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 480 ]), &(acadoWorkspace.evGu[ 240 ]), &(acadoWorkspace.x[ 60 ]), &(acadoWorkspace.sbar[ 120 ]), &(acadoWorkspace.sbar[ 124 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 496 ]), &(acadoWorkspace.evGu[ 248 ]), &(acadoWorkspace.x[ 62 ]), &(acadoWorkspace.sbar[ 124 ]), &(acadoWorkspace.sbar[ 128 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 512 ]), &(acadoWorkspace.evGu[ 256 ]), &(acadoWorkspace.x[ 64 ]), &(acadoWorkspace.sbar[ 128 ]), &(acadoWorkspace.sbar[ 132 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 528 ]), &(acadoWorkspace.evGu[ 264 ]), &(acadoWorkspace.x[ 66 ]), &(acadoWorkspace.sbar[ 132 ]), &(acadoWorkspace.sbar[ 136 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 544 ]), &(acadoWorkspace.evGu[ 272 ]), &(acadoWorkspace.x[ 68 ]), &(acadoWorkspace.sbar[ 136 ]), &(acadoWorkspace.sbar[ 140 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 560 ]), &(acadoWorkspace.evGu[ 280 ]), &(acadoWorkspace.x[ 70 ]), &(acadoWorkspace.sbar[ 140 ]), &(acadoWorkspace.sbar[ 144 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 576 ]), &(acadoWorkspace.evGu[ 288 ]), &(acadoWorkspace.x[ 72 ]), &(acadoWorkspace.sbar[ 144 ]), &(acadoWorkspace.sbar[ 148 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 592 ]), &(acadoWorkspace.evGu[ 296 ]), &(acadoWorkspace.x[ 74 ]), &(acadoWorkspace.sbar[ 148 ]), &(acadoWorkspace.sbar[ 152 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 608 ]), &(acadoWorkspace.evGu[ 304 ]), &(acadoWorkspace.x[ 76 ]), &(acadoWorkspace.sbar[ 152 ]), &(acadoWorkspace.sbar[ 156 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 624 ]), &(acadoWorkspace.evGu[ 312 ]), &(acadoWorkspace.x[ 78 ]), &(acadoWorkspace.sbar[ 156 ]), &(acadoWorkspace.sbar[ 160 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 640 ]), &(acadoWorkspace.evGu[ 320 ]), &(acadoWorkspace.x[ 80 ]), &(acadoWorkspace.sbar[ 160 ]), &(acadoWorkspace.sbar[ 164 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 656 ]), &(acadoWorkspace.evGu[ 328 ]), &(acadoWorkspace.x[ 82 ]), &(acadoWorkspace.sbar[ 164 ]), &(acadoWorkspace.sbar[ 168 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 672 ]), &(acadoWorkspace.evGu[ 336 ]), &(acadoWorkspace.x[ 84 ]), &(acadoWorkspace.sbar[ 168 ]), &(acadoWorkspace.sbar[ 172 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 688 ]), &(acadoWorkspace.evGu[ 344 ]), &(acadoWorkspace.x[ 86 ]), &(acadoWorkspace.sbar[ 172 ]), &(acadoWorkspace.sbar[ 176 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 704 ]), &(acadoWorkspace.evGu[ 352 ]), &(acadoWorkspace.x[ 88 ]), &(acadoWorkspace.sbar[ 176 ]), &(acadoWorkspace.sbar[ 180 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 720 ]), &(acadoWorkspace.evGu[ 360 ]), &(acadoWorkspace.x[ 90 ]), &(acadoWorkspace.sbar[ 180 ]), &(acadoWorkspace.sbar[ 184 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 736 ]), &(acadoWorkspace.evGu[ 368 ]), &(acadoWorkspace.x[ 92 ]), &(acadoWorkspace.sbar[ 184 ]), &(acadoWorkspace.sbar[ 188 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 752 ]), &(acadoWorkspace.evGu[ 376 ]), &(acadoWorkspace.x[ 94 ]), &(acadoWorkspace.sbar[ 188 ]), &(acadoWorkspace.sbar[ 192 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 768 ]), &(acadoWorkspace.evGu[ 384 ]), &(acadoWorkspace.x[ 96 ]), &(acadoWorkspace.sbar[ 192 ]), &(acadoWorkspace.sbar[ 196 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 784 ]), &(acadoWorkspace.evGu[ 392 ]), &(acadoWorkspace.x[ 98 ]), &(acadoWorkspace.sbar[ 196 ]), &(acadoWorkspace.sbar[ 200 ]) );
for (lRun1 = 0; lRun1 < 204; ++lRun1)
acadoVariables.x[lRun1] += acadoWorkspace.sbar[lRun1];

}

int acado_preparationStep(  )
{
int ret;

ret = acado_modelSimulation();
acado_evaluateObjective(  );
acado_condensePrep(  );
return ret;
}

int acado_feedbackStep(  )
{
int tmp;

acado_condenseFdb(  );

tmp = acado_solve( );

acado_expand(  );
return tmp;
}

int acado_initializeSolver(  )
{
int ret;

/* This is a function which must be called once before any other function call! */


ret = 0;

memset(&acadoWorkspace, 0, sizeof( acadoWorkspace ));
acadoVariables.lbValues[0] = -2.0000000000000001e-01;
acadoVariables.lbValues[1] = -5.0000000000000000e-01;
acadoVariables.lbValues[2] = -2.0000000000000001e-01;
acadoVariables.lbValues[3] = -5.0000000000000000e-01;
acadoVariables.lbValues[4] = -2.0000000000000001e-01;
acadoVariables.lbValues[5] = -5.0000000000000000e-01;
acadoVariables.lbValues[6] = -2.0000000000000001e-01;
acadoVariables.lbValues[7] = -5.0000000000000000e-01;
acadoVariables.lbValues[8] = -2.0000000000000001e-01;
acadoVariables.lbValues[9] = -5.0000000000000000e-01;
acadoVariables.lbValues[10] = -2.0000000000000001e-01;
acadoVariables.lbValues[11] = -5.0000000000000000e-01;
acadoVariables.lbValues[12] = -2.0000000000000001e-01;
acadoVariables.lbValues[13] = -5.0000000000000000e-01;
acadoVariables.lbValues[14] = -2.0000000000000001e-01;
acadoVariables.lbValues[15] = -5.0000000000000000e-01;
acadoVariables.lbValues[16] = -2.0000000000000001e-01;
acadoVariables.lbValues[17] = -5.0000000000000000e-01;
acadoVariables.lbValues[18] = -2.0000000000000001e-01;
acadoVariables.lbValues[19] = -5.0000000000000000e-01;
acadoVariables.lbValues[20] = -2.0000000000000001e-01;
acadoVariables.lbValues[21] = -5.0000000000000000e-01;
acadoVariables.lbValues[22] = -2.0000000000000001e-01;
acadoVariables.lbValues[23] = -5.0000000000000000e-01;
acadoVariables.lbValues[24] = -2.0000000000000001e-01;
acadoVariables.lbValues[25] = -5.0000000000000000e-01;
acadoVariables.lbValues[26] = -2.0000000000000001e-01;
acadoVariables.lbValues[27] = -5.0000000000000000e-01;
acadoVariables.lbValues[28] = -2.0000000000000001e-01;
acadoVariables.lbValues[29] = -5.0000000000000000e-01;
acadoVariables.lbValues[30] = -2.0000000000000001e-01;
acadoVariables.lbValues[31] = -5.0000000000000000e-01;
acadoVariables.lbValues[32] = -2.0000000000000001e-01;
acadoVariables.lbValues[33] = -5.0000000000000000e-01;
acadoVariables.lbValues[34] = -2.0000000000000001e-01;
acadoVariables.lbValues[35] = -5.0000000000000000e-01;
acadoVariables.lbValues[36] = -2.0000000000000001e-01;
acadoVariables.lbValues[37] = -5.0000000000000000e-01;
acadoVariables.lbValues[38] = -2.0000000000000001e-01;
acadoVariables.lbValues[39] = -5.0000000000000000e-01;
acadoVariables.lbValues[40] = -2.0000000000000001e-01;
acadoVariables.lbValues[41] = -5.0000000000000000e-01;
acadoVariables.lbValues[42] = -2.0000000000000001e-01;
acadoVariables.lbValues[43] = -5.0000000000000000e-01;
acadoVariables.lbValues[44] = -2.0000000000000001e-01;
acadoVariables.lbValues[45] = -5.0000000000000000e-01;
acadoVariables.lbValues[46] = -2.0000000000000001e-01;
acadoVariables.lbValues[47] = -5.0000000000000000e-01;
acadoVariables.lbValues[48] = -2.0000000000000001e-01;
acadoVariables.lbValues[49] = -5.0000000000000000e-01;
acadoVariables.lbValues[50] = -2.0000000000000001e-01;
acadoVariables.lbValues[51] = -5.0000000000000000e-01;
acadoVariables.lbValues[52] = -2.0000000000000001e-01;
acadoVariables.lbValues[53] = -5.0000000000000000e-01;
acadoVariables.lbValues[54] = -2.0000000000000001e-01;
acadoVariables.lbValues[55] = -5.0000000000000000e-01;
acadoVariables.lbValues[56] = -2.0000000000000001e-01;
acadoVariables.lbValues[57] = -5.0000000000000000e-01;
acadoVariables.lbValues[58] = -2.0000000000000001e-01;
acadoVariables.lbValues[59] = -5.0000000000000000e-01;
acadoVariables.lbValues[60] = -2.0000000000000001e-01;
acadoVariables.lbValues[61] = -5.0000000000000000e-01;
acadoVariables.lbValues[62] = -2.0000000000000001e-01;
acadoVariables.lbValues[63] = -5.0000000000000000e-01;
acadoVariables.lbValues[64] = -2.0000000000000001e-01;
acadoVariables.lbValues[65] = -5.0000000000000000e-01;
acadoVariables.lbValues[66] = -2.0000000000000001e-01;
acadoVariables.lbValues[67] = -5.0000000000000000e-01;
acadoVariables.lbValues[68] = -2.0000000000000001e-01;
acadoVariables.lbValues[69] = -5.0000000000000000e-01;
acadoVariables.lbValues[70] = -2.0000000000000001e-01;
acadoVariables.lbValues[71] = -5.0000000000000000e-01;
acadoVariables.lbValues[72] = -2.0000000000000001e-01;
acadoVariables.lbValues[73] = -5.0000000000000000e-01;
acadoVariables.lbValues[74] = -2.0000000000000001e-01;
acadoVariables.lbValues[75] = -5.0000000000000000e-01;
acadoVariables.lbValues[76] = -2.0000000000000001e-01;
acadoVariables.lbValues[77] = -5.0000000000000000e-01;
acadoVariables.lbValues[78] = -2.0000000000000001e-01;
acadoVariables.lbValues[79] = -5.0000000000000000e-01;
acadoVariables.lbValues[80] = -2.0000000000000001e-01;
acadoVariables.lbValues[81] = -5.0000000000000000e-01;
acadoVariables.lbValues[82] = -2.0000000000000001e-01;
acadoVariables.lbValues[83] = -5.0000000000000000e-01;
acadoVariables.lbValues[84] = -2.0000000000000001e-01;
acadoVariables.lbValues[85] = -5.0000000000000000e-01;
acadoVariables.lbValues[86] = -2.0000000000000001e-01;
acadoVariables.lbValues[87] = -5.0000000000000000e-01;
acadoVariables.lbValues[88] = -2.0000000000000001e-01;
acadoVariables.lbValues[89] = -5.0000000000000000e-01;
acadoVariables.lbValues[90] = -2.0000000000000001e-01;
acadoVariables.lbValues[91] = -5.0000000000000000e-01;
acadoVariables.lbValues[92] = -2.0000000000000001e-01;
acadoVariables.lbValues[93] = -5.0000000000000000e-01;
acadoVariables.lbValues[94] = -2.0000000000000001e-01;
acadoVariables.lbValues[95] = -5.0000000000000000e-01;
acadoVariables.lbValues[96] = -2.0000000000000001e-01;
acadoVariables.lbValues[97] = -5.0000000000000000e-01;
acadoVariables.lbValues[98] = -2.0000000000000001e-01;
acadoVariables.lbValues[99] = -5.0000000000000000e-01;
acadoVariables.ubValues[0] = 2.0000000000000001e-01;
acadoVariables.ubValues[1] = 5.0000000000000000e-01;
acadoVariables.ubValues[2] = 2.0000000000000001e-01;
acadoVariables.ubValues[3] = 5.0000000000000000e-01;
acadoVariables.ubValues[4] = 2.0000000000000001e-01;
acadoVariables.ubValues[5] = 5.0000000000000000e-01;
acadoVariables.ubValues[6] = 2.0000000000000001e-01;
acadoVariables.ubValues[7] = 5.0000000000000000e-01;
acadoVariables.ubValues[8] = 2.0000000000000001e-01;
acadoVariables.ubValues[9] = 5.0000000000000000e-01;
acadoVariables.ubValues[10] = 2.0000000000000001e-01;
acadoVariables.ubValues[11] = 5.0000000000000000e-01;
acadoVariables.ubValues[12] = 2.0000000000000001e-01;
acadoVariables.ubValues[13] = 5.0000000000000000e-01;
acadoVariables.ubValues[14] = 2.0000000000000001e-01;
acadoVariables.ubValues[15] = 5.0000000000000000e-01;
acadoVariables.ubValues[16] = 2.0000000000000001e-01;
acadoVariables.ubValues[17] = 5.0000000000000000e-01;
acadoVariables.ubValues[18] = 2.0000000000000001e-01;
acadoVariables.ubValues[19] = 5.0000000000000000e-01;
acadoVariables.ubValues[20] = 2.0000000000000001e-01;
acadoVariables.ubValues[21] = 5.0000000000000000e-01;
acadoVariables.ubValues[22] = 2.0000000000000001e-01;
acadoVariables.ubValues[23] = 5.0000000000000000e-01;
acadoVariables.ubValues[24] = 2.0000000000000001e-01;
acadoVariables.ubValues[25] = 5.0000000000000000e-01;
acadoVariables.ubValues[26] = 2.0000000000000001e-01;
acadoVariables.ubValues[27] = 5.0000000000000000e-01;
acadoVariables.ubValues[28] = 2.0000000000000001e-01;
acadoVariables.ubValues[29] = 5.0000000000000000e-01;
acadoVariables.ubValues[30] = 2.0000000000000001e-01;
acadoVariables.ubValues[31] = 5.0000000000000000e-01;
acadoVariables.ubValues[32] = 2.0000000000000001e-01;
acadoVariables.ubValues[33] = 5.0000000000000000e-01;
acadoVariables.ubValues[34] = 2.0000000000000001e-01;
acadoVariables.ubValues[35] = 5.0000000000000000e-01;
acadoVariables.ubValues[36] = 2.0000000000000001e-01;
acadoVariables.ubValues[37] = 5.0000000000000000e-01;
acadoVariables.ubValues[38] = 2.0000000000000001e-01;
acadoVariables.ubValues[39] = 5.0000000000000000e-01;
acadoVariables.ubValues[40] = 2.0000000000000001e-01;
acadoVariables.ubValues[41] = 5.0000000000000000e-01;
acadoVariables.ubValues[42] = 2.0000000000000001e-01;
acadoVariables.ubValues[43] = 5.0000000000000000e-01;
acadoVariables.ubValues[44] = 2.0000000000000001e-01;
acadoVariables.ubValues[45] = 5.0000000000000000e-01;
acadoVariables.ubValues[46] = 2.0000000000000001e-01;
acadoVariables.ubValues[47] = 5.0000000000000000e-01;
acadoVariables.ubValues[48] = 2.0000000000000001e-01;
acadoVariables.ubValues[49] = 5.0000000000000000e-01;
acadoVariables.ubValues[50] = 2.0000000000000001e-01;
acadoVariables.ubValues[51] = 5.0000000000000000e-01;
acadoVariables.ubValues[52] = 2.0000000000000001e-01;
acadoVariables.ubValues[53] = 5.0000000000000000e-01;
acadoVariables.ubValues[54] = 2.0000000000000001e-01;
acadoVariables.ubValues[55] = 5.0000000000000000e-01;
acadoVariables.ubValues[56] = 2.0000000000000001e-01;
acadoVariables.ubValues[57] = 5.0000000000000000e-01;
acadoVariables.ubValues[58] = 2.0000000000000001e-01;
acadoVariables.ubValues[59] = 5.0000000000000000e-01;
acadoVariables.ubValues[60] = 2.0000000000000001e-01;
acadoVariables.ubValues[61] = 5.0000000000000000e-01;
acadoVariables.ubValues[62] = 2.0000000000000001e-01;
acadoVariables.ubValues[63] = 5.0000000000000000e-01;
acadoVariables.ubValues[64] = 2.0000000000000001e-01;
acadoVariables.ubValues[65] = 5.0000000000000000e-01;
acadoVariables.ubValues[66] = 2.0000000000000001e-01;
acadoVariables.ubValues[67] = 5.0000000000000000e-01;
acadoVariables.ubValues[68] = 2.0000000000000001e-01;
acadoVariables.ubValues[69] = 5.0000000000000000e-01;
acadoVariables.ubValues[70] = 2.0000000000000001e-01;
acadoVariables.ubValues[71] = 5.0000000000000000e-01;
acadoVariables.ubValues[72] = 2.0000000000000001e-01;
acadoVariables.ubValues[73] = 5.0000000000000000e-01;
acadoVariables.ubValues[74] = 2.0000000000000001e-01;
acadoVariables.ubValues[75] = 5.0000000000000000e-01;
acadoVariables.ubValues[76] = 2.0000000000000001e-01;
acadoVariables.ubValues[77] = 5.0000000000000000e-01;
acadoVariables.ubValues[78] = 2.0000000000000001e-01;
acadoVariables.ubValues[79] = 5.0000000000000000e-01;
acadoVariables.ubValues[80] = 2.0000000000000001e-01;
acadoVariables.ubValues[81] = 5.0000000000000000e-01;
acadoVariables.ubValues[82] = 2.0000000000000001e-01;
acadoVariables.ubValues[83] = 5.0000000000000000e-01;
acadoVariables.ubValues[84] = 2.0000000000000001e-01;
acadoVariables.ubValues[85] = 5.0000000000000000e-01;
acadoVariables.ubValues[86] = 2.0000000000000001e-01;
acadoVariables.ubValues[87] = 5.0000000000000000e-01;
acadoVariables.ubValues[88] = 2.0000000000000001e-01;
acadoVariables.ubValues[89] = 5.0000000000000000e-01;
acadoVariables.ubValues[90] = 2.0000000000000001e-01;
acadoVariables.ubValues[91] = 5.0000000000000000e-01;
acadoVariables.ubValues[92] = 2.0000000000000001e-01;
acadoVariables.ubValues[93] = 5.0000000000000000e-01;
acadoVariables.ubValues[94] = 2.0000000000000001e-01;
acadoVariables.ubValues[95] = 5.0000000000000000e-01;
acadoVariables.ubValues[96] = 2.0000000000000001e-01;
acadoVariables.ubValues[97] = 5.0000000000000000e-01;
acadoVariables.ubValues[98] = 2.0000000000000001e-01;
acadoVariables.ubValues[99] = 5.0000000000000000e-01;
return ret;
}

void acado_initializeNodesByForwardSimulation(  )
{
int index;
for (index = 0; index < 50; ++index)
{
state[0] = acadoVariables.x[index * 4];
state[1] = acadoVariables.x[index * 4 + 1];
state[2] = acadoVariables.x[index * 4 + 2];
state[3] = acadoVariables.x[index * 4 + 3];
state[28] = acadoVariables.u[index * 2];
state[29] = acadoVariables.u[index * 2 + 1];

acado_integrate(state, index == 0);

acadoVariables.x[index * 4 + 4] = state[0];
acadoVariables.x[index * 4 + 5] = state[1];
acadoVariables.x[index * 4 + 6] = state[2];
acadoVariables.x[index * 4 + 7] = state[3];
}
}

void acado_shiftStates( int strategy, real_t* const xEnd, real_t* const uEnd )
{
int index;
for (index = 0; index < 50; ++index)
{
acadoVariables.x[index * 4] = acadoVariables.x[index * 4 + 4];
acadoVariables.x[index * 4 + 1] = acadoVariables.x[index * 4 + 5];
acadoVariables.x[index * 4 + 2] = acadoVariables.x[index * 4 + 6];
acadoVariables.x[index * 4 + 3] = acadoVariables.x[index * 4 + 7];
}

if (strategy == 1 && xEnd != 0)
{
acadoVariables.x[200] = xEnd[0];
acadoVariables.x[201] = xEnd[1];
acadoVariables.x[202] = xEnd[2];
acadoVariables.x[203] = xEnd[3];
}
else if (strategy == 2) 
{
state[0] = acadoVariables.x[200];
state[1] = acadoVariables.x[201];
state[2] = acadoVariables.x[202];
state[3] = acadoVariables.x[203];
if (uEnd != 0)
{
state[28] = uEnd[0];
state[29] = uEnd[1];
}
else
{
state[28] = acadoVariables.u[98];
state[29] = acadoVariables.u[99];
}

acado_integrate(state, 1);

acadoVariables.x[200] = state[0];
acadoVariables.x[201] = state[1];
acadoVariables.x[202] = state[2];
acadoVariables.x[203] = state[3];
}
}

void acado_shiftControls( real_t* const uEnd )
{
int index;
for (index = 0; index < 49; ++index)
{
acadoVariables.u[index * 2] = acadoVariables.u[index * 2 + 2];
acadoVariables.u[index * 2 + 1] = acadoVariables.u[index * 2 + 3];
}

if (uEnd != 0)
{
acadoVariables.u[98] = uEnd[0];
acadoVariables.u[99] = uEnd[1];
}
}

real_t acado_getKKT(  )
{
real_t kkt;

int index;
real_t prd;

kkt = + acadoWorkspace.g[0]*acadoWorkspace.x[0] + acadoWorkspace.g[1]*acadoWorkspace.x[1] + acadoWorkspace.g[2]*acadoWorkspace.x[2] + acadoWorkspace.g[3]*acadoWorkspace.x[3] + acadoWorkspace.g[4]*acadoWorkspace.x[4] + acadoWorkspace.g[5]*acadoWorkspace.x[5] + acadoWorkspace.g[6]*acadoWorkspace.x[6] + acadoWorkspace.g[7]*acadoWorkspace.x[7] + acadoWorkspace.g[8]*acadoWorkspace.x[8] + acadoWorkspace.g[9]*acadoWorkspace.x[9] + acadoWorkspace.g[10]*acadoWorkspace.x[10] + acadoWorkspace.g[11]*acadoWorkspace.x[11] + acadoWorkspace.g[12]*acadoWorkspace.x[12] + acadoWorkspace.g[13]*acadoWorkspace.x[13] + acadoWorkspace.g[14]*acadoWorkspace.x[14] + acadoWorkspace.g[15]*acadoWorkspace.x[15] + acadoWorkspace.g[16]*acadoWorkspace.x[16] + acadoWorkspace.g[17]*acadoWorkspace.x[17] + acadoWorkspace.g[18]*acadoWorkspace.x[18] + acadoWorkspace.g[19]*acadoWorkspace.x[19] + acadoWorkspace.g[20]*acadoWorkspace.x[20] + acadoWorkspace.g[21]*acadoWorkspace.x[21] + acadoWorkspace.g[22]*acadoWorkspace.x[22] + acadoWorkspace.g[23]*acadoWorkspace.x[23] + acadoWorkspace.g[24]*acadoWorkspace.x[24] + acadoWorkspace.g[25]*acadoWorkspace.x[25] + acadoWorkspace.g[26]*acadoWorkspace.x[26] + acadoWorkspace.g[27]*acadoWorkspace.x[27] + acadoWorkspace.g[28]*acadoWorkspace.x[28] + acadoWorkspace.g[29]*acadoWorkspace.x[29] + acadoWorkspace.g[30]*acadoWorkspace.x[30] + acadoWorkspace.g[31]*acadoWorkspace.x[31] + acadoWorkspace.g[32]*acadoWorkspace.x[32] + acadoWorkspace.g[33]*acadoWorkspace.x[33] + acadoWorkspace.g[34]*acadoWorkspace.x[34] + acadoWorkspace.g[35]*acadoWorkspace.x[35] + acadoWorkspace.g[36]*acadoWorkspace.x[36] + acadoWorkspace.g[37]*acadoWorkspace.x[37] + acadoWorkspace.g[38]*acadoWorkspace.x[38] + acadoWorkspace.g[39]*acadoWorkspace.x[39] + acadoWorkspace.g[40]*acadoWorkspace.x[40] + acadoWorkspace.g[41]*acadoWorkspace.x[41] + acadoWorkspace.g[42]*acadoWorkspace.x[42] + acadoWorkspace.g[43]*acadoWorkspace.x[43] + acadoWorkspace.g[44]*acadoWorkspace.x[44] + acadoWorkspace.g[45]*acadoWorkspace.x[45] + acadoWorkspace.g[46]*acadoWorkspace.x[46] + acadoWorkspace.g[47]*acadoWorkspace.x[47] + acadoWorkspace.g[48]*acadoWorkspace.x[48] + acadoWorkspace.g[49]*acadoWorkspace.x[49] + acadoWorkspace.g[50]*acadoWorkspace.x[50] + acadoWorkspace.g[51]*acadoWorkspace.x[51] + acadoWorkspace.g[52]*acadoWorkspace.x[52] + acadoWorkspace.g[53]*acadoWorkspace.x[53] + acadoWorkspace.g[54]*acadoWorkspace.x[54] + acadoWorkspace.g[55]*acadoWorkspace.x[55] + acadoWorkspace.g[56]*acadoWorkspace.x[56] + acadoWorkspace.g[57]*acadoWorkspace.x[57] + acadoWorkspace.g[58]*acadoWorkspace.x[58] + acadoWorkspace.g[59]*acadoWorkspace.x[59] + acadoWorkspace.g[60]*acadoWorkspace.x[60] + acadoWorkspace.g[61]*acadoWorkspace.x[61] + acadoWorkspace.g[62]*acadoWorkspace.x[62] + acadoWorkspace.g[63]*acadoWorkspace.x[63] + acadoWorkspace.g[64]*acadoWorkspace.x[64] + acadoWorkspace.g[65]*acadoWorkspace.x[65] + acadoWorkspace.g[66]*acadoWorkspace.x[66] + acadoWorkspace.g[67]*acadoWorkspace.x[67] + acadoWorkspace.g[68]*acadoWorkspace.x[68] + acadoWorkspace.g[69]*acadoWorkspace.x[69] + acadoWorkspace.g[70]*acadoWorkspace.x[70] + acadoWorkspace.g[71]*acadoWorkspace.x[71] + acadoWorkspace.g[72]*acadoWorkspace.x[72] + acadoWorkspace.g[73]*acadoWorkspace.x[73] + acadoWorkspace.g[74]*acadoWorkspace.x[74] + acadoWorkspace.g[75]*acadoWorkspace.x[75] + acadoWorkspace.g[76]*acadoWorkspace.x[76] + acadoWorkspace.g[77]*acadoWorkspace.x[77] + acadoWorkspace.g[78]*acadoWorkspace.x[78] + acadoWorkspace.g[79]*acadoWorkspace.x[79] + acadoWorkspace.g[80]*acadoWorkspace.x[80] + acadoWorkspace.g[81]*acadoWorkspace.x[81] + acadoWorkspace.g[82]*acadoWorkspace.x[82] + acadoWorkspace.g[83]*acadoWorkspace.x[83] + acadoWorkspace.g[84]*acadoWorkspace.x[84] + acadoWorkspace.g[85]*acadoWorkspace.x[85] + acadoWorkspace.g[86]*acadoWorkspace.x[86] + acadoWorkspace.g[87]*acadoWorkspace.x[87] + acadoWorkspace.g[88]*acadoWorkspace.x[88] + acadoWorkspace.g[89]*acadoWorkspace.x[89] + acadoWorkspace.g[90]*acadoWorkspace.x[90] + acadoWorkspace.g[91]*acadoWorkspace.x[91] + acadoWorkspace.g[92]*acadoWorkspace.x[92] + acadoWorkspace.g[93]*acadoWorkspace.x[93] + acadoWorkspace.g[94]*acadoWorkspace.x[94] + acadoWorkspace.g[95]*acadoWorkspace.x[95] + acadoWorkspace.g[96]*acadoWorkspace.x[96] + acadoWorkspace.g[97]*acadoWorkspace.x[97] + acadoWorkspace.g[98]*acadoWorkspace.x[98] + acadoWorkspace.g[99]*acadoWorkspace.x[99];
kkt = fabs( kkt );
for (index = 0; index < 100; ++index)
{
prd = acadoWorkspace.y[index];
if (prd > 1e-12)
kkt += fabs(acadoWorkspace.lb[index] * prd);
else if (prd < -1e-12)
kkt += fabs(acadoWorkspace.ub[index] * prd);
}
return kkt;
}

real_t acado_getObjective(  )
{
real_t objVal;

int lRun1;
/** Row vector of size: 4 */
real_t tmpDy[ 4 ];

/** Row vector of size: 4 */
real_t tmpDyN[ 4 ];

for (lRun1 = 0; lRun1 < 50; ++lRun1)
{
acadoWorkspace.objValueIn[0] = acadoVariables.x[lRun1 * 4];
acadoWorkspace.objValueIn[1] = acadoVariables.x[lRun1 * 4 + 1];
acadoWorkspace.objValueIn[2] = acadoVariables.x[lRun1 * 4 + 2];
acadoWorkspace.objValueIn[3] = acadoVariables.x[lRun1 * 4 + 3];
acadoWorkspace.objValueIn[4] = acadoVariables.u[lRun1 * 2];
acadoWorkspace.objValueIn[5] = acadoVariables.u[lRun1 * 2 + 1];

acado_evaluateLSQ( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.Dy[lRun1 * 4] = acadoWorkspace.objValueOut[0] - acadoVariables.y[lRun1 * 4];
acadoWorkspace.Dy[lRun1 * 4 + 1] = acadoWorkspace.objValueOut[1] - acadoVariables.y[lRun1 * 4 + 1];
acadoWorkspace.Dy[lRun1 * 4 + 2] = acadoWorkspace.objValueOut[2] - acadoVariables.y[lRun1 * 4 + 2];
acadoWorkspace.Dy[lRun1 * 4 + 3] = acadoWorkspace.objValueOut[3] - acadoVariables.y[lRun1 * 4 + 3];
}
acadoWorkspace.objValueIn[0] = acadoVariables.x[200];
acadoWorkspace.objValueIn[1] = acadoVariables.x[201];
acadoWorkspace.objValueIn[2] = acadoVariables.x[202];
acadoWorkspace.objValueIn[3] = acadoVariables.x[203];
acado_evaluateLSQEndTerm( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.DyN[0] = acadoWorkspace.objValueOut[0] - acadoVariables.yN[0];
acadoWorkspace.DyN[1] = acadoWorkspace.objValueOut[1] - acadoVariables.yN[1];
acadoWorkspace.DyN[2] = acadoWorkspace.objValueOut[2] - acadoVariables.yN[2];
acadoWorkspace.DyN[3] = acadoWorkspace.objValueOut[3] - acadoVariables.yN[3];
objVal = 0.0000000000000000e+00;
for (lRun1 = 0; lRun1 < 50; ++lRun1)
{
tmpDy[0] = + acadoWorkspace.Dy[lRun1 * 4]*acadoVariables.W[lRun1 * 16] + acadoWorkspace.Dy[lRun1 * 4 + 1]*acadoVariables.W[lRun1 * 16 + 4] + acadoWorkspace.Dy[lRun1 * 4 + 2]*acadoVariables.W[lRun1 * 16 + 8] + acadoWorkspace.Dy[lRun1 * 4 + 3]*acadoVariables.W[lRun1 * 16 + 12];
tmpDy[1] = + acadoWorkspace.Dy[lRun1 * 4]*acadoVariables.W[lRun1 * 16 + 1] + acadoWorkspace.Dy[lRun1 * 4 + 1]*acadoVariables.W[lRun1 * 16 + 5] + acadoWorkspace.Dy[lRun1 * 4 + 2]*acadoVariables.W[lRun1 * 16 + 9] + acadoWorkspace.Dy[lRun1 * 4 + 3]*acadoVariables.W[lRun1 * 16 + 13];
tmpDy[2] = + acadoWorkspace.Dy[lRun1 * 4]*acadoVariables.W[lRun1 * 16 + 2] + acadoWorkspace.Dy[lRun1 * 4 + 1]*acadoVariables.W[lRun1 * 16 + 6] + acadoWorkspace.Dy[lRun1 * 4 + 2]*acadoVariables.W[lRun1 * 16 + 10] + acadoWorkspace.Dy[lRun1 * 4 + 3]*acadoVariables.W[lRun1 * 16 + 14];
tmpDy[3] = + acadoWorkspace.Dy[lRun1 * 4]*acadoVariables.W[lRun1 * 16 + 3] + acadoWorkspace.Dy[lRun1 * 4 + 1]*acadoVariables.W[lRun1 * 16 + 7] + acadoWorkspace.Dy[lRun1 * 4 + 2]*acadoVariables.W[lRun1 * 16 + 11] + acadoWorkspace.Dy[lRun1 * 4 + 3]*acadoVariables.W[lRun1 * 16 + 15];
objVal += + acadoWorkspace.Dy[lRun1 * 4]*tmpDy[0] + acadoWorkspace.Dy[lRun1 * 4 + 1]*tmpDy[1] + acadoWorkspace.Dy[lRun1 * 4 + 2]*tmpDy[2] + acadoWorkspace.Dy[lRun1 * 4 + 3]*tmpDy[3];
}

tmpDyN[0] = + acadoWorkspace.DyN[0]*acadoVariables.WN[0];
tmpDyN[1] = + acadoWorkspace.DyN[1]*acadoVariables.WN[5];
tmpDyN[2] = + acadoWorkspace.DyN[2]*acadoVariables.WN[10];
tmpDyN[3] = + acadoWorkspace.DyN[3]*acadoVariables.WN[15];
objVal += + acadoWorkspace.DyN[0]*tmpDyN[0] + acadoWorkspace.DyN[1]*tmpDyN[1] + acadoWorkspace.DyN[2]*tmpDyN[2] + acadoWorkspace.DyN[3]*tmpDyN[3];

objVal *= 0.5;
return objVal;
}

