#!/bin/bash

echo "Running std_eeg+video_inter_cross.py..."
nohup python std_eeg+video_inter_cross.py > std_inter_cross.log 2>&1 &
PID1=$!
wait $PID1
echo "std_eeg+video_inter_cross.py finished."

echo "Running std_eeg+video_inter_concat.py..."
nohup python std_eeg+video_inter_concat.py > std_inter_concat.log 2>&1 &
PID2=$!
wait $PID2
echo "std_eeg+video_inter_concat.py finished."

echo "Running std_eeg+video_inter_intra.py..."
nohup python std_eeg+video_inter_intra.py > std_inter_intra.log 2>&1 &
PID3=$!
wait $PID3
echo "std_eeg+video_inter_intra.py finished."

echo "✅ 실행 완료. 로그 확인: std_inter_cross.log, std_inter_concat.log, std_inter_intra.log"
