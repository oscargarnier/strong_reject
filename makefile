alloc:
	salloc --account=c2016159 \
	--constraint=MI250 \
	--job-name=dynamic_allocation_qwen \
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

mini:
	python run_attack_and_evaluation.py \
		--attack pair \
		--victim_model llama2-7 \
		--dataset strong_reject3 \
		--evaluator strongreject_finetuned \
		--internet_free_api \
	 	--api_queue_dir api_queue

api_worker:
	python scripts/file_api_worker.py \
		--queue-root api_queue \
		--job-name testing_job \
		--job-id 12345 \
		--env-file .env
