# code generator build script

set -e

cd ~/Project/motionplanning_ws/ACADOtoolkit/build
make
cd ~/Project/motionplanning_ws/ACADOtoolkit/examples/getting_started/
./simple_mpc
cd ~/Project/motionplanning_ws/ACADOtoolkit/examples/getting_started/simple_mpc_export
cp acado_aux*.h ~/Project/motionplanning_ws/acado/mpc_acado
cp acado_aux*.c ~/Project/motionplanning_ws/acado/mpc_acado
cp acado_common.h ~/Project/motionplanning_ws/acado/mpc_acado
cp acado_integrator.c ~/Project/motionplanning_ws/acado/mpc_acado
cp acado_qp* ~/Project/motionplanning_ws/acado/mpc_acado
cp acado_solver.c ~/Project/motionplanning_ws/acado/mpc_acado
cd ~/Project/motionplanning_ws/acado/mpc_acado
sudo python3 setup.py build install --force
cd ~/Project/motionplanning_ws/acado/


