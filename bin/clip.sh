#!/bin/bash

AVISYNTH="/mnt/c/Program Files (x86)/VirtualDub-1.10.4/vdub.exe"

"$AVISYNTH" /c /s submission.vcf /p clip.avs clip.avi /r /x
