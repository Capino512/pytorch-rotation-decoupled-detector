

void _overlaps(float* overlaps,const float* boxes,const float* query_boxes, int n, int k, int device_id);


void _rotate_nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, int device_id);
