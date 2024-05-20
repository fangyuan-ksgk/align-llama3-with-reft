pip install  --upgrade transformers datasets accelerate evaluate bitsandbytes trl peft torch
pip install ninja packaging tensorboardX sentencepiece
pip install --upgrade openai pyreft
MAX_JOBS=4 pip install flash-attn --no-build-isolation
pip install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes
pip install git+https://github.com/stanfordnlp/pyreft.git
pip install dataclasses