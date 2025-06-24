#!/bin/bash

echo "Running mult_final.py..."
nohup python mult_final.py > mult_final.log 2>&1 && \
echo "mult_final.py finished. Running mult_final2.py..." && \
nohup python mult_final2.py > mult_final2.log 2>&1 &

echo "✅ 실행 완료. 로그 확인: mult_final.log, mult_final2.log"
