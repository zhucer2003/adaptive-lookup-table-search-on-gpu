__kernel void interpolation(
          int N, 
          __global float* value_x,
          __global float *value_y, 
          __global int* index_g, 
          __global int *level_list_d, 
          __global float *centerx_list_d, 
          __global float* centery_list_d,
          __global float *T1_list_d, 
          __global float* T2_list_d, 
          __global float * T3_list_d, 
          __global float* T4_list_d,
          __global float* interp_value){

    float xmin, ymin, xmax, ymax, width;
    int i = get_local_id(0);
    if(i< N){
	    int j = index_g[i];
	    width = pow(2.0,-level_list_d[j]*1.0);
	    xmin = centerx_list_d[j] - width;
	    ymin = centery_list_d[j] - width;
	    xmax = centerx_list_d[j] + width;
	    ymax = centery_list_d[j] + width; 

	    // rescale x,y in the local cell
	    float x_ref = (value_x[i]-xmin)/(xmax-xmin);
	    float y_ref = (value_y[i]-ymin)/(ymax-xmin);
	   
	    // pickup the interpolation triangle 
            float x_nodes[3], y_nodes[3], var[3];
	    x_nodes[0] = xmin;
	    x_nodes[1] = xmax ;
	    x_nodes[2] = x_ref>=y_ref?  xmax: xmin;
	    
            y_nodes[0] = ymin;
	    y_nodes[1] = x_ref>=y_ref? ymin:ymax ;
	    y_nodes[2] = ymax ;
	    
            var[0] = T1_list_d[j];
	    var[1] = x_ref>=y_ref? T2_list_d[j]: T3_list_d[j] ;
	    var[2] = x_ref>=y_ref? T3_list_d[j]: T4_list_d[j];
	float A = y_nodes[0]*(var[1]- var[2])  
                  +  y_nodes[1]*(var[2] - var[0]) 
                  +  y_nodes[2]*(var[0] - var[1]);

	float B = var[0]*(x_nodes[1] - x_nodes[2])
                   + var[1]*(x_nodes[2] - x_nodes[0])
                   +  var[2]*(x_nodes[0] - x_nodes[1]);

	float C = x_nodes[0]*(y_nodes[1] - y_nodes[2])
                  + x_nodes[1]*(y_nodes[2] - y_nodes[0])
                  + x_nodes[2]*(y_nodes[0] - y_nodes[1]);

	float D = -A*x_nodes[0] - B*y_nodes[0] - C*var[0];
	interp_value[i] = -(A*value_x[i] + B*value_y[i] + D)/C;
   }
}
