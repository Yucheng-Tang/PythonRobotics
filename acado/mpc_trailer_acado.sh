# code generator build script

set -e

#cd ~/Project/motionplanning_ws/ACADOtoolkit/build
#make
#cd ~/Project/motionplanning_ws/ACADOtoolkit/examples/getting_started/
#./trailer_mpc
cd ~/projects/research/motionplanning/ACADOtoolkit/examples/getting_started/petral_mpc_export
cp acado_aux*.h ~/projects/research/motionplanning/PythonRobotics/acado/petra_mpc_obstacle_acado
cp acado_aux*.c ~/projects/research/motionplanning/PythonRobotics/acado/petra_mpc_obstacle_acado
cp acado_common.h ~/projects/research/motionplanning/PythonRobotics/acado/petra_mpc_obstacle_acado
cp acado_integrator.c ~/projects/research/motionplanning/PythonRobotics/acado/petra_mpc_obstacle_acado
cp acado_qp* ~/projects/research/motionplanning/PythonRobotics/acado/petra_mpc_obstacle_acado
cp acado_solver.c ~/projects/research/motionplanning/PythonRobotics/acado/petra_mpc_obstacle_acado
cd ~/projects/research/motionplanning/PythonRobotics/acado/petra_mpc_obstacle_acado
# sudo python3 setup.py build install --force
# cd ~/Project/motionplanning_ws/acado/

cd ~/projects/research/motionplanning/ACADOtoolkit/examples/getting_started/simple_mpc_export
cp acado_aux*.h ~/projects/research/motionplanning/PythonRobotics/acado/mpc_acado
cp acado_aux*.c ~/projects/research/motionplanning/PythonRobotics/acado/mpc_acado
cp acado_common.h ~/projects/research/motionplanning/PythonRobotics/acado/mpc_acado
cp acado_integrator.c ~/projects/research/motionplanning/PythonRobotics/acado/mpc_acado
cp acado_qp* ~/projects/research/motionplanning/PythonRobotics/acado/mpc_acado
cp acado_solver.c ~/projects/research/motionplanning/PythonRobotics/acado/mpc_acado
cd ~/projects/research/motionplanning/PythonRobotics/acado/mpc_acado
