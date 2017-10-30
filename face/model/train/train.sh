DIR=512_100
cd ${DIR}
echo "enter ${DIR} run caffe"
nohup caffe train -solver solver.prototxt -log_dir log -gpu all >/dev/null 2>&1 &
cd -
echo "exit ${DIR}"

DIR=512_200
cd ${DIR}
echo "enter ${DIR} run caffe"
nohup caffe train -solver solver.prototxt -log_dir log -gpu all >/dev/null 2>&1 &
cd -
echo "exit ${DIR}"

DIR=1024_100
cd ${DIR}
echo "enter ${DIR} run caffe"
nohup caffe train -solver solver.prototxt -log_dir log -gpu all >/dev/null 2>&1 &
cd -
echo "exit ${DIR}"

DIR=1024_200
cd ${DIR}
echo "enter ${DIR} run caffe"
nohup caffe train -solver solver.prototxt -log_dir log -gpu all >/dev/null 2>&1 &
cd -
echo "exit ${DIR}"
