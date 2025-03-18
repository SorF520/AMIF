
for num_epochs in '20'
do
	for lr in '1e-5'
	do
		for warmup_ratio in '0.2'
		do
			for seed in '2023'
			do
				for batch_size in '32'
				do
					for max_seq in '64'
					do
					  for alpha in '0.5'
					  do
					    for margin in '0.1'
					    do
					      for SGR_step in '3'
					      do
					        for weight_js in '0.8'
					        do
                    echo ${num_epochs}
                    echo ${lr}
                    echo ${warmup_ratio}
                    echo ${seed}
                    echo ${batch_size}
                    echo ${max_seq}
                    echo ${alpha}
                    echo ${margin}
                    echo ${SGR_step}
                    echo ${weight_js}
                    CUDA_VISIBLE_DEVICES=0 python run.py  \
                    --num_epochs ${num_epochs} \
                    --lr ${lr} \
                    --warmup_ratio ${warmup_ratio} \
                    --seed ${seed} \
                    --batch_size ${batch_size} \
                    --max_seq ${max_seq} \
                    --alpha ${alpha} \
                    --margin ${margin} \
                    --SGR_step ${SGR_step} \
                    --weight_js ${weight_js}
                  done
                done
              done
						done
					done
				done
			done
		done
	done
done
