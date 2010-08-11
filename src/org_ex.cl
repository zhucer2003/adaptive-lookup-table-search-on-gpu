void kernel square(global int *data) {
  int id = get_global_id(0);
  data[id] *= data[id];
}


void kernel search_kernel(global int len,global int N,global  float *value_x,global  float* value_y,global  int *index,global  int *level_list_d,global  int* leaf_list_d,global  float* centerx_list_d,global  float *centery_list_d){
    
    float xmin, ymin, xmax, ymax, width; 
    int = get_global_id(0);
    if (i<len){
       width = pow(2.0,-level_list_d[i]);
       xmin = centerx_list_d[i] - width;
       ymin = centery_list_d[i] - width;
       xmax = centerx_list_d[i] + width;
       ymax = centery_list_d[i] + width;
    }
    const int s_width=1024;
    __local float x_loc[s_width], y_loc[s_width];
    
    for (int m = 0;m<N/s_width +1 ;m++)
    {   
        loc_id = get_local_id(0); 
        int k = m*s_width + threadIdx.x;
        if(k<N && threadIdx.x<s_width){
         x_loc[threadIdx.x] = value_x[m*s_width + threadIdx.x];
         y_loc[threadIdx.x] = value_y[m*s_width + threadIdx.x];
        }
      barrier(CLK_LOCAL_MEM_FENCE);
        if (i< len){
           for (int j=0;j<s_width ; j++){
           if (x_loc[j] >= xmin && x_loc[j]<=xmax &&
               y_loc[j] > ymin && y_loc[j]<=ymax )
              index[j+m*s_width] = i;    
            }

        }
      barrier(CLK_LOCAL_MEM_FENCE);
      }
}
