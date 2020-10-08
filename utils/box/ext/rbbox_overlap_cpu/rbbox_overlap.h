

#include <cmath>


# define PI 3.14159265358979323846


double trangle_area(double * a, double * b, double * c) {
  return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]))/2.0;
}


double area(double * int_pts, int num_of_inter) {

  double area = 0;
  for(int i = 0;i < num_of_inter - 2;i++) {
    area += fabs(trangle_area(int_pts, int_pts + 2 * i + 2, int_pts + 2 * i + 4));
  }
  return area;
}


void reorder_pts(double * int_pts, int num_of_inter) {

  if(num_of_inter > 0) {

    double center[2];

    center[0] = 0.0;
    center[1] = 0.0;

    for(int i = 0;i < num_of_inter;i++) {
      center[0] += int_pts[2 * i];
      center[1] += int_pts[2 * i + 1];
    }
    center[0] /= num_of_inter;
    center[1] /= num_of_inter;

    double vs[16];
    double v[2];
    double d;
    for(int i = 0;i < num_of_inter;i++) {
      v[0] = int_pts[2 * i]-center[0];
      v[1] = int_pts[2 * i + 1]-center[1];
      d = sqrt(v[0] * v[0] + v[1] * v[1]);
      v[0] = v[0] / d;
      v[1] = v[1] / d;
      if(v[1] < 0) {
        v[0]= - 2 - v[0];
      }
      vs[i] = v[0];
    }

    double temp,tx,ty;
    int j;
    for(int i=1;i<num_of_inter;++i){
      if(vs[i-1]>vs[i]){
        temp = vs[i];
        tx = int_pts[2*i];
        ty = int_pts[2*i+1];
        j=i;
        while(j>0&&vs[j-1]>temp){
          vs[j] = vs[j-1];
          int_pts[j*2] = int_pts[j*2-2];
          int_pts[j*2+1] = int_pts[j*2-1];
          j--;
        }
        vs[j] = temp;
        int_pts[j*2] = tx;
        int_pts[j*2+1] = ty;
      }
    }
  }
}

bool inter2line(double * pts1, double *pts2, int i, int j, double * temp_pts) {

  double a[2];
  double b[2];
  double c[2];
  double d[2];

  double area_abc, area_abd, area_cda, area_cdb;

  a[0] = pts1[2 * i];
  a[1] = pts1[2 * i + 1];

  b[0] = pts1[2 * ((i + 1) % 4)];
  b[1] = pts1[2 * ((i + 1) % 4) + 1];

  c[0] = pts2[2 * j];
  c[1] = pts2[2 * j + 1];

  d[0] = pts2[2 * ((j + 1) % 4)];
  d[1] = pts2[2 * ((j + 1) % 4) + 1];

  area_abc = trangle_area(a, b, c);
  area_abd = trangle_area(a, b, d);

  if(area_abc * area_abd >= 0) {
    return false;
  }

  area_cda = trangle_area(c, d, a);
  area_cdb = area_cda + area_abc - area_abd;

  if (area_cda * area_cdb >= 0) {
    return false;
  }
  double t = area_cda / (area_abd - area_abc);

  double dx = t * (b[0] - a[0]);
  double dy = t * (b[1] - a[1]);
  temp_pts[0] = a[0] + dx;
  temp_pts[1] = a[1] + dy;

  return true;
}

bool in_rect(double pt_x, double pt_y, double * pts) {

  double ab[2];
  double ad[2];
  double ap[2];

  double abab;
  double abap;
  double adad;
  double adap;

  ab[0] = pts[2] - pts[0];
  ab[1] = pts[3] - pts[1];

  ad[0] = pts[6] - pts[0];
  ad[1] = pts[7] - pts[1];

  ap[0] = pt_x - pts[0];
  ap[1] = pt_y - pts[1];

  abab = ab[0] * ab[0] + ab[1] * ab[1];
  abap = ab[0] * ap[0] + ab[1] * ap[1];
  adad = ad[0] * ad[0] + ad[1] * ad[1];
  adap = ad[0] * ap[0] + ad[1] * ap[1];

  return abab >= abap && abap >= 0 && adad >= adap && adap >= 0;
}

int inter_pts(double * pts1, double * pts2, double * int_pts) {

  int num_of_inter = 0;

  for(int i = 0;i < 4;i++) {
    if(in_rect(pts1[2 * i], pts1[2 * i + 1], pts2)) {
      int_pts[num_of_inter * 2] = pts1[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1];
      num_of_inter++;
    }
     if(in_rect(pts2[2 * i], pts2[2 * i + 1], pts1)) {
      int_pts[num_of_inter * 2] = pts2[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1];
      num_of_inter++;
    }
  }

  double temp_pts[2];

  for(int i = 0;i < 4;i++) {
    for(int j = 0;j < 4;j++) {
      bool has_pts = inter2line(pts1, pts2, i, j, temp_pts);
      if(has_pts) {
        int_pts[num_of_inter * 2] = temp_pts[0];
        int_pts[num_of_inter * 2 + 1] = temp_pts[1];
        num_of_inter++;
      }
    }
  }


  return num_of_inter;
}


void convert_region(double * pts , double * region) {

  double angle = region[4];
  double a_cos = cos(angle/180.0*PI);
  double a_sin = sin(angle/180.0*PI);

  double ctr_x = region[0];
  double ctr_y = region[1];

  double w = region[2];
  double h = region[3];

  double pts_x[4];
  double pts_y[4];

  pts_x[0] = - w / 2;
  pts_x[1] = w / 2;
  pts_x[2] = w / 2;
  pts_x[3] = - w / 2;

  pts_y[0] = - h / 2;
  pts_y[1] = - h / 2;
  pts_y[2] = h / 2;
  pts_y[3] = h / 2;

  for(int i = 0;i < 4;i++) {
    pts[7 - 2 * i - 1] = a_cos * pts_x[i] - a_sin * pts_y[i] + ctr_x;
    pts[7 - 2 * i] = a_sin * pts_x[i] + a_cos * pts_y[i] + ctr_y;

  }

}


double inter(double * region1, double * region2) {

  double pts1[8];
  double pts2[8];
  double int_pts[16];
  int num_of_inter;

  convert_region(pts1, region1);
  convert_region(pts2, region2);

  num_of_inter = inter_pts(pts1, pts2, int_pts);

  reorder_pts(int_pts, num_of_inter);

  return area(int_pts, num_of_inter);


}

double RotateIoU(double * region1, double * region2) {

  double area1 = region1[2] * region1[3];
  double area2 = region2[2] * region2[3];
  double area_inter = inter(region1, region2);

  return area_inter / (area1 + area2 - area_inter);

}


void RotateIoU_1x1(double * region1, double * region2, int n, double * ret){
    for ( int i = 0; i < n; i++ ){
        ret[i] = RotateIoU(region1 + i * 5, region2 + i * 5);
    }
}


void RotateIoU_nxn(double * region1, double * region2, int n1, int n2, double * ret){
    for ( int i = 0; i < n1; i++ ){
        for ( int j = 0; j < n2; j++ ){
            ret[i * n2 + j] = RotateIoU(region1 + i * 5, region2 + j * 5);
        }
    }
}

void RotateNMS(double * bboxes, int n, double thresh, int * keeps){
    int i, flag;
    n--;
    while(n > 0){
        flag = 0;
        for ( i = 0; i < n; i++ ){
            if (keeps[i]){
                if (RotateIoU(bboxes + n * 5, bboxes + i * 5) > thresh){
                    keeps[i] = 0;
                }
                else{
                    flag = i;
                }
            }
        }
        n = flag;
    }
}