# comp9318


		data_blocks = split_data(data)
		max_iter = 20
		loop for max_iter:
			for each data_block in data_blocks:    
				for each centroid_block in initial_cantroid:
					# 求K_means(data_block,centroid_block)
					all_cluster = []
					for each line in data_block:
						nearest_centroid = min_distance(line, centroid_block)
						# all_cluster are supposed to be (N,1)
						all_cluster.append(nearest_centroid)
					for each centroid in centroid_block:
						all_data = cluster_data(centroid,all_cluster)
						if len(all_data) > 0:
							# 求all_data 在每个维度的上中位数
							centroid = update_centroid(all_data)
						else:
							pass
							
		 # now we get new_centroid = initial_centroid = (p,k,m/p) = codebooks
		 code = []
		 for each line_data in data:
		 	 nearest_centroid = min_distance(line_data, codebooks)
		 	 code.append(nearest_centroid)
		 return codebook,code
		  		
		 
		 
					
					
			
			
	
		
		
	
	



