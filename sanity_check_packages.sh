# 1. Syntax check: does your code at least parse/compile?
python -m compileall .

# 2. Do the imports resolve?
python -c "import strong_reject; print('strong_reject ok')"
python -c "import torch; print(torch.__version__, torch.version.hip)"

# 3. Dependency consistency
pip check

# 4. Nothing got swapped
pip list | grep -i -E "torch|nvidia|triton"
