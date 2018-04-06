#!/bin/bash

for speaker in {1..34}
	do
		base="http://spandh.dcs.shef.ac.uk/gridcorpus/s"
		ext="/video/s"
		file=".mpg_vcd.zip"
		
		url=$base$speaker$ext$speaker$file
		wget $url

		zipfile="s"$speaker$file
		unzip $zipfile
		rm $zipfile

		mkdir "s"$speaker"/video"
		mv "s"$speaker/*.mpg "s"$speaker"/video/"

		cd "s"$speaker
		wget $base$speaker"/align/s"$speaker".tar"
		tar xopf "s"$speaker".tar"
		rm "s"$speaker".tar"

		cd ../
	done
