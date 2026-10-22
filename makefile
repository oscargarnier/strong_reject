alloc:
	salloc --account=c2016159 \
       --constraint=MI250 \
       --job-name=gcg_transfer_harmbench \
       --ntasks=1 \
       --nodes=1 \
       --gres=gpu:1 \
       --mem=32G \
       --cpus-per-task=4 \
       --threads-per-core=1 \
       --time=01:00:00
