#!/usr/bin/env bash
# Patch use_fast=True cho AutoProcessor (workaround SigLIP-2 SiglipTokenizer bug).
cd /root/bd

# Patch encoder.py
python3 -c "
import re
p = 'app/common/encoder.py'
s = open(p).read()
s = s.replace(
    'AutoProcessor.from_pretrained(model_name)',
    'AutoProcessor.from_pretrained(model_name, use_fast=True)'
)
open(p, 'w').write(s)
print('encoder.py patched')
"

# Patch remote_download_models.py
python3 -c "
p = 'scripts/remote_download_models.py'
s = open(p).read()
s = s.replace(
    'AutoProcessor.from_pretrained(\"google/siglip2-base-patch16-384\")',
    'AutoProcessor.from_pretrained(\"google/siglip2-base-patch16-384\", use_fast=True)'
)
open(p, 'w').write(s)
print('remote_download_models.py patched')
"

grep use_fast app/common/encoder.py scripts/remote_download_models.py
