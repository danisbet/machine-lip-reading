#!/bin/bash
trap "exit" INT
for speaker in {1..34}
	do
		python extract_mouth_batch.py "../data/s"$speaker"/video/" "*.mpg" target/ shape_predictor_68_face_landmarks.dat
	done

