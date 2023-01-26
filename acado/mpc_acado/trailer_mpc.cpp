/*
 *    This file is part of ACADO Toolkit.
 *
 *    ACADO Toolkit -- A Toolkit for Automatic Control and Dynamic Optimization.
 *    Copyright (C) 2008-2014 by Boris Houska, Hans Joachim Ferreau,
 *    Milan Vukov, Rien Quirynen, KU Leuven.
 *    Developed within the Optimization in Engineering Center (OPTEC)
 *    under supervision of Moritz Diehl. All rights reserved.
 *
 *    ACADO Toolkit is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    ACADO Toolkit is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with ACADO Toolkit; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

 /**
 *    \file   examples/code_generation/petral_mpc.cpp
 *    \author Hans Joachim Ferreau, Boris Houska
 *    \date   2011-2013
 */

#include <acado_code_generation.hpp>

int main( )
{
	USING_NAMESPACE_ACADO

	const bool CODE_GEN = true;

	// Variables:
	DifferentialState   x    ;  // the x position
	DifferentialState   y    ;  // the y velocity
	DifferentialState   yaw  ;  // the robot angle
	DifferentialState   yawt ;  // the trailer angle
	Control             v    ;  // the velocity
    Control             w    ;  // the angular velocity

    IntermediateState   f = x**2    ;

    Stream              s    ;  // debugging test
    s << f;

	const double     g = 9.81;  // the gravitational constant
    const double     l = 0.5;   // the rod length
	const double     cp_offset = 0.36; // the coupling offset

	// Model equations:
	DifferentialEquation f;

	f << dot( x ) == v * cos(yaw - yawt) * cos(yawt);
	f << dot( y ) == v * cos(yaw - yawt) * sin(yawt);
	f << dot( yaw ) == w;
    f << dot( yawt ) == (v / l * sin(yaw - yawt) - cp_offset * w / l * cos(yaw - yawt));

	// Reference functions and weighting matrices:
	Function h, hN;
	h << x << y << yaw << yawt << v << w;
	hN << x << y << yaw << yawt;

	// Running cost weight matrix
	DMatrix Q(h.getDim(), h.getDim());
	Q.setIdentity();
	Q(0,0) = 100;
	Q(1,1) = 100;
	Q(2,2) = 10;
	Q(3,3) = 100;
	Q(4,4) = 100;
	Q(5,5) = 100;

	// End cost weight matrix
	DMatrix QN(hN.getDim(), hN.getDim());
  	QN.setIdentity();
	QN(0,0) = Q(0,0);
	QN(1,1) = Q(1,1);
	QN(2,2) = Q(2,2);
	QN(3,3) = Q(3,3);

	DVector r(h.getDim());    // Running cost reference
	DVector rN(hN.getDim());   // End cost reference
	r.setZero();
	rN.setZero();

	// WN *= 5;

	//
	// Optimal Control Problem
	//
	const double t_start = 0.0;
	const double t_end   = 1.0;

	OCP ocp(t_start, t_end, 50);

	if(!CODE_GEN)
	{
		// For analysis, set references.
		ocp.minimizeLSQ( Q, h, r );
		ocp.minimizeLSQEndTerm( QN, hN, rN );
	}else{
		// For code generation, references are set during run time.
		BMatrix Q_sparse(h.getDim(), h.getDim());
		Q_sparse.setIdentity();
		BMatrix QN_sparse(hN.getDim(), hN.getDim());
		QN_sparse.setIdentity();
		ocp.minimizeLSQ( Q_sparse, h );
		ocp.minimizeLSQEndTerm( QN_sparse, hN );
	}

	double max_vel = 0.2;
	double max_angvel = 0.5;
	ocp.subjectTo( f );
	ocp.subjectTo( -max_vel <= v <= max_vel );
	ocp.subjectTo( -max_angvel <= w <= max_angvel );

	// Export the code:
	OCPexport mpc( ocp );

	mpc.set( HESSIAN_APPROXIMATION,       GAUSS_NEWTON    );
	mpc.set( DISCRETIZATION_TYPE,         MULTIPLE_SHOOTING );
    mpc.set(SPARSE_QP_SOLUTION,           FULL_CONDENSING_N2);  // due to qpOASES
	mpc.set( INTEGRATOR_TYPE,             INT_RK4         );
	mpc.set( NUM_INTEGRATOR_STEPS,        200              );
	mpc.set( QP_SOLVER,                   QP_QPOASES      );

    mpc.set(HOTSTART_QP,            YES);
	mpc.set(CG_USE_OPENMP,                    YES);       // paralellization
    mpc.set(CG_HARDCODE_CONSTRAINT_VALUES,    NO );       // set on runtime
    mpc.set(CG_USE_VARIABLE_WEIGHTING_MATRIX, YES);       // time-varying costs
    mpc.set( USE_SINGLE_PRECISION,            YES);       // Single precision

    mpc.set( GENERATE_TEST_FILE,          NO);
    mpc.set( GENERATE_MAKE_FILE,          NO);
    mpc.set( GENERATE_MATLAB_INTERFACE,   NO);
    mpc.set( GENERATE_SIMULINK_INTERFACE, NO);

// 	mpc.set( USE_SINGLE_PRECISION,        YES             );

	if (mpc.exportCode( "petral_mpc_export" ) != SUCCESSFUL_RETURN)
		exit( EXIT_FAILURE );

	mpc.printDimensionsQP( );

	return EXIT_SUCCESS;
}
