alloc:
	salloc --account=c2016159 \
	--constraint=MI250 \
	--job-name=internet_test_pair \
	--ntasks=1 \
	--nodes=1 \
	--gres=gpu:1 \
	--mem=32G \
	--cpus-per-task=4 \
	--threads-per-core=1 \
	--time=01:00:00

small_alloc:
	salloc --account=c2016159 \
	--constraint=MI250 \
	--job-name=dynamic_allocation_small \
	--ntasks=1 \
	--nodes=1 \
	--gres=gpu:1 \
	--mem=16G \
	--cpus-per-task=4 \
	--threads-per-core=1 \
	--time=01:00:00

launch_attacks:
	while read -r atk; do sbatch slurm_scripts/$${atk}_sample.sbatch; done < slurm_scripts/current_attacks.txt
uncensored_sample:
	while read -r model; do sbatch slurm_scripts/$${model}_strong_reject10_uncensored.sbatch; done < slurm_scripts/uncensored_models.txt
	
uncensored_fullrun:
	while read -r model; do sbatch slurm_scripts/$${model}_strong_reject313_uncensored.sbatch; done < slurm_scripts/uncensored_models.txt

CURRENT_JOB := pap_evaluation_test

launch_script:
	sbatch slurm_scripts/$(CURRENT_JOB).sbatch

api_worker:
	/usr/bin/time -v python scripts/file_api_worker.py \
		--queue-root api_queue \
		--job-name $(CURRENT_JOB) \
		--job-id $(ID) \
		--env-file .env
