

dt=$(date +'%Y%m%d')

activate project-nursena

python -u notebook_$dt.py 2>&1 | tee notebook_$dt.log

mkdir -p results-$dt

mv *.pdf results-$dt
mv *.log results-$dt
