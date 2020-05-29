
// This is a ray tracer implemented by Yuxuan Huang for CSCI 5607

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <math.h>
#include <limits>
#include <random>
#include <stack>

#define PI 3.14159265

using namespace std;

/* 3d Vector struct */
struct vec3 {
	float x;
	float y;
	float z;

	vec3() {
		x = 0;
		y = 0;
		z = 0;
	}

	vec3(float a, float b, float c) {
		x = a;
		y = b;
		z = c;
	}

	vec3 operator+(const vec3 &v2) const{
		vec3 result;
		result.x = x + v2.x;
		result.y = y + v2.y;
		result.z = z + v2.z;
		return result;
	}

	vec3 operator-() const{
		vec3 result;
		result.x = -x;
		result.y = -y;
		result.z = -z;
		return result;
	}

	vec3 operator-(const vec3 &v2) const{
		vec3 result;
		result.x = x - v2.x;
		result.y = y - v2.y;
		result.z = z - v2.z;
		return result;
	}

	vec3 operator*(float a) const{
		vec3 result;
		result.x = a * x;
		result.y = a * y;
		result.z = a * z;
		return result;
	}

	friend vec3 operator*(float a, vec3 v2);

	float operator*(const vec3& v2) {
		return x * v2.x + y * v2.y + z * v2.z;
	}

	vec3 cross(const vec3& v2) const{
		vec3 result;
		result.x = y * v2.z - z * v2.y;
		result.y = z * v2.x - x * v2.z;
		result.z = x * v2.y - y * v2.x;
		return result;
	}

	bool operator==(const vec3& v2) {
		return(x == v2.x && y == v2.y && z == v2.z);
	}

	void operator=(const vec3& v2) {
		x = v2.x;
		y = v2.y;
		z = v2.z;
	}

	float length() {
		return sqrt(x * x + y * y + z * z);
	}

	void normalize() {
		float l = length();
		x /= l;
		y /= l;
		z /= l;
	}
};

vec3 operator*(float a, vec3 v2) {
	vec3 result;
	result.x = a * v2.x;
	result.y = a * v2.y;
	result.z = a * v2.z;
	return result;
}

vector<vec3> VE; // array of vertices
vector<vec3> VN; // array of vertex normals
vector<vec3> VT; // array of textures indices

/* Read File Functions */
// function to remove the extraneous
void trim(string& s) {
	if (!s.empty()) {
		s.erase(s.find_last_not_of(" ") + 1);
	}
}

// function to read file into a string
string readfile(string filename) {
	ifstream spec;
	string content;
	string result;
	spec.open(filename);
	if (!spec) { // Handling the exception when the file cannot be found
		cout << "Cannot find " << filename << endl;
		exit(1);
	}
	while (getline(spec, content)) { // Read all the lines of the file
		result += " ";
		result += content;
	}
	trim(result);
	return result;
}


/* Classes */
bool depthofview = false;
float D = 1; // viewing distance

class viewing_prm {
public:
	vec3 eye;
	vec3 viewdir;
	vec3 updir;
	float vfov;
	float img_h;
	float img_w;

	// check the validity of the viewing parameters
	// and normalize them along the way
	bool isvalid() {
		if (vfov >= 180.0 || vfov <= 0) return false;
		if (img_h <= 0 || img_w <= 0) return false;
		viewdir.normalize();
		updir.normalize();
		if (viewdir == updir || viewdir == -updir) return false;
		return true;
	}
};

class mtlcolor {
public:
	vec3 d_color; // diffuse color
	vec3 s_color; // specular color
	vec3 k; // ka kd and ks
	float n; // power of the specular term
	float alpha; // opacity
	float eta; // index of refraction
	float F0; // Fresnel 

	// diffuse and specular color has to be between 0 and 1
	// so are the ks
	bool isvalid() {
		if (d_color.x < 0 || d_color.y < 0 || d_color.z < 0) return false;
		if (d_color.x > 1 || d_color.y > 1 || d_color.z > 1) return false;
		if (s_color.x < 0 || s_color.y < 0 || s_color.z < 0) return false;
		if (s_color.x > 1 || s_color.y > 1 || s_color.z > 1) return false;
		if (k.x < 0 || k.y < 0 || k.z < 0) return false;
		if (k.x > 1 || k.y > 1 || k.z > 1) return false;
		if (n < 1 || eta < 0) return false;
		if (alpha > 1 || alpha < 0) return false;
		F0 = ((eta - 1) / (eta + 1)) * ((eta - 1) / (eta + 1));
		return true;
	}
};

vector<mtlcolor> M_COLOR; // array of material color

class texture {
public:
	vec3 **t_img; // texture image (maximum 4k resolution)
	int width;
	int height;

	texture(string texturefile) {
		t_img = new vec3 * [3000];
		for (int i = 0; i < 3000; i++) {
			t_img[i] = new vec3[3000];
		}
		ifstream t;
		t.open(texturefile);

		if (!t) { // Handling the exception when the file cannot be found
			cout << "Cannot find " << texturefile << endl;
			exit(1);
		}
		else {
			stringstream ss;
			string content;
			string colorinfo;
			string p;
			int maxcolor;
			// max color ignored for now
			getline(t, content); // header info
			ss.str(content);
			ss >> p >> width >> height >> maxcolor;
			ss.clear();
			while (getline(t, content)) {
				colorinfo += " ";
				colorinfo += content;
			}
			ss.str(colorinfo);
			for (int h = 0; h < height; h++) {
				for (int w = 0; w < width; w++) {
					ss >> t_img[h][w].x >> t_img[h][w].y >> t_img[h][w].z;
					//if (h == height-1) cout << t_img[h][w].x << " " << t_img[h][w].y << " " << t_img[h][w].z << endl;
					t_img[h][w].x /= maxcolor;
					t_img[h][w].y /= maxcolor;
					t_img[h][w].z /= maxcolor;
				}
			}
			ss.clear();
			//cout << t_img[0][0].x << " " << t_img[0][0].y << " " << t_img[0][0].z << endl;
		}
	}
	
	/*~texture() {
		for (int i = 0; i < 3000; i++) {
			delete[] t_img[i];
		}
		delete[] t_img;
	}*/
};

vector<texture> TT; // array of texture;

class sphereobj {
public:
	mtlcolor mtlclr;
	vec3 position;
	int t; //texture index
	float r;

	sphereobj() {
		t = -1; // default no texture
	}

	/*check the validity of the object*/
	bool isvalid() {
		if (r <= 0) return false;
		return true;
	}

};

vector<int> transphere; // array of the indices of transparent spheres
vector<int> opaqsphere; // array of the indices of opaque spheres

/*
class ellipsoidobj {
public:
	mtlcolor mtlclr;
	vector<float> ellipsoid;

	ellipsoidobj() {
		vector<float> f6(6, 0.0);
		ellipsoid = f6;
	}

	bool isvalid() {
		if (ellipsoid[3] * ellipsoid[4] * ellipsoid[5] <= 0) return false;
		return true;
	}
};
*/

class lightobj {
public:
	vec3 pod; // position or direction
	float flag;
	vec3 color;
	vec3 c; // for attenuation
	vec3 dir; // for spotlight direction
	float theta; // for spotlight cut-off value

	lightobj() {
		flag = 1; // default point light
		c.x = 1; // default no attenuation
		theta = 0; // default not spotlight
	}

	bool isvalid() {
		if (flag != 0 && flag != 1) return false;
		if (c.x < 0 || c.y < 0 || c.z < 0) return false;
		if (theta < 0 || theta > 180) return false;
		if (color.x < 0 || color.y < 0 || color.z < 0) return false;
		if (color.x > 1 || color.y > 1 || color.z > 1) return false;
		return true;
	}
};

class face {
public:
	int vertices[3]; // index of vertices
	int texture[3]; // index of texture coordinates
	int vnormal[3]; // index of vertex normals
	int mc_index; // index of material color
	int t_index; // index of texture image
	vec3 normal;
	bool texture_pvd;
	bool normal_pvd;

	face() {
		texture_pvd = false; // texture provided
		normal_pvd = false;  // normal provided
	}

	void comp_normal() {
		normal = (VE[vertices[1] - 1] - VE[vertices[0] - 1]).\
			cross((VE[vertices[2] - 1] - VE[vertices[0] - 1]));
		normal.normalize();
	}
};

vector<int> transface; // array of the indices of transparent faces
vector<int> opaqface; // array of the indices of opaque faces

void readface(face& f, string s[]) {
	if (s[0].find("/") == std::string::npos) { // no normal or texture provided
		for (int i = 0; i < 3; i++) {
			f.vertices[i] = stoi(s[i]);
		}
		f.comp_normal();
	}
	else {
		int pos1 = s[0].find("/");
		int pos2 = s[0].rfind("/");
		if (pos1 == pos2) { // no normal provided
			for (int i = 0; i < 3; i++) {
				pos1 = s[i].find("/");
				pos2 = s[i].rfind("/");
				f.vertices[i] = stoi(s[i].substr(0, pos1));
				f.texture[i] = stoi(s[i].substr(pos1 + 1, s[i].length() - pos1 - 1));
			}
			f.texture_pvd = true;
			f.comp_normal();
		}
		else if (pos2 == pos1 + 1) { // no texture provided
			for(int i = 0; i < 3; i++) {
				pos1 = s[i].find("/");
				pos2 = s[i].rfind("/");
				f.vertices[i] = stoi(s[i].substr(0, pos1));
				f.vnormal[i] = stoi(s[i].substr(pos2 + 1, s[i].length() - pos2 - 1));
			}
			f.normal_pvd = true;
		}
		else {
			for (int i = 0; i < 3; i++) {
				pos1 = s[i].find("/");
				pos2 = s[i].rfind("/");
				f.vertices[i] = stoi(s[i].substr(0, pos1));
				f.texture[i] = stoi(s[i].substr(pos1 + 1, pos2 - pos1 - 1));
				f.vnormal[i] = stoi(s[i].substr(pos2 + 1, s[i].length() - pos2 - 1));
			}
			f.normal_pvd = true;
			f.texture_pvd = true;
		}
	}
}

/* Ray Tacing Functions */
enum raytype {inside, outside}; // rays that are inside or outside a surface

stack<float> ETA;

// function to compute the direction of rays (normalized)
void computevw(const viewing_prm& v, vec3& ul, vec3& vstep, vec3& hstep) {
	vec3 u = v.viewdir.cross(v.updir);
	u.normalize();
	vec3 up = u.cross(v.viewdir);
	up.normalize();
	vec3 center = v.eye + D * v.viewdir; // center of the viewing window
	float vhd = tan(v.vfov / 2 * PI / 180.0) * D; // vertical half distance
	float hhd = vhd * v.img_w / v.img_h; // horizontal half distance

	ul = center + vhd * up - hhd * u; // upper left
	vec3 ur = center + vhd * up + hhd * u; // upper right
	vec3 ll = center - vhd * up - hhd * u; // lower left
	vec3 lr = center - vhd * up + hhd * u; // lower right

	// vertical and horizontal step
	vstep = (1.0 / (v.img_h - 1)) * (ll - ul);
	hstep = (1.0 / (v.img_w - 1)) * (ur - ul);
}

void computerays(const viewing_prm& v, vec3& ul, vec3& vstep, vec3& hstep,\
vector<vector<vec3>>& rays) {
	for (int i = 0; i < v.img_h; i++) { // compute the direction of each eay
		for (int j = 0; j < v.img_w; j++) {
			rays[i][j] = ul + i * vstep + j * hstep - v.eye;
			rays[i][j].normalize();
		}
	}
}

// function to draw varying background
void drawbkg(const vec3& bkgcolor, vector<vector<vec3>>& img, \
	const viewing_prm& v) {
	vec3 baseclr(1, 1, 1);
	vec3 inc = (1.0 / (v.img_h - 1))* (baseclr - bkgcolor);
	for (int i = 0; i < v.img_h; i++) {
		vec3 color = bkgcolor + float(i) * inc;
		for (int j = 0; j < v.img_w; j++) {
			img[i][j] = color;
		}
	}
}

// function to compute intersections using determinant
// and return the closest intersection
float comp_intersec(float a, float b, float c) {
	float delta = b * b - 4 * a * c;
	if (delta >= 0) {
		float t1 = (-b - sqrt(delta)) / (2 * a);
		float t2 = (-b + sqrt(delta)) / (2 * a);
		if (t1 > 0) return t1;
		else if (t2 > 0) return t2;
	}
	return numeric_limits<float>::max();
}


// function to compute intersections with sphere (compute a, b and c)
float intersect_s(const vec3& ray, const vec3& org, const sphereobj& s) {

	float xc = s.position.x;
	float yc = s.position.y;
	float zc = s.position.z;
	float r = s.r;
	float a = 1.0;
	float c = (org.x - xc) * (org.x - xc) + (org.y - yc) * (org.y - yc) \
		+ (org.z - zc) * (org.z - zc) - r * r;
	float b = 2 * (ray.x * (org.x - xc) + \
		ray.y * (org.y - yc) + ray.z * (org.z - zc));
	return comp_intersec(a, b, c);
}

vec3 s_get_texture(const texture& t, const vec3& N, const sphereobj& s) {
	vec3 texture_color;
	float v = acos(N.z) / PI;
	float theta = atan2(N.y, N.x);
	float u;
	if (theta < 0)  theta += (2 * PI);
	u = theta / (2 * PI);
	texture_color = t.t_img[int(round(v * (t.height - 1)))][int(round(u * (t.width - 1)))];
	return texture_color;
}

// determine whether an intersection is within a triangle
// and return a, b, c
bool in_triangle(const vec3& p, const vec3& p0, const vec3& p1, const vec3& p2,\
	const face& f, float bc_coord[]) {
	float epsilon = 0.001;
	vec3 e0 = p - p0;
	vec3 e1 = p1 - p;
	vec3 e2 = p2 - p;
	vec3 e01 = p1 - p0;
	vec3 e02 = p2 - p0;
	float A = e01.cross(e02).length();
	float a = e1.cross(e2).length()/A;
	float b = e0.cross(e02).length()/A;
	float c = e01.cross(e0).length()/A;
	if (a + b + c - 1 < epsilon) { // in triangle
		bc_coord[0] = a;
		bc_coord[1] = b;
		bc_coord[2] = c;
		return true;
	}
	return false;
}

// function to compute intersections with triangles
float intersect_t(const face& f, const vec3& ray,\
	const vec3& org, float bc_coord[]) {
	vec3 p0 = VE[f.vertices[0] - 1];
	vec3 p1 = VE[f.vertices[1] - 1];
	vec3 p2 = VE[f.vertices[2] - 1];
	vec3 e1 = p1 - p0;
	vec3 e2 = p2 - p0;
	vec3 n = e1.cross(e2);
	float D = -(n.x * p0.x + n.y * p0.y + n.z * p0.z);
	float den = n.x * ray.x + n.y * ray.y + n.z * ray.z;
	if (den == 0) return numeric_limits<float>::max(); // parallel
	float nom = -(n.x * org.x + n.y * org.y + n.z * org.z + D);
	float t = nom / den;
	if (t < 0) return numeric_limits<float>::max(); // intersection behind the eye
	else {
		vec3 intersec = org + t * ray;
		if (in_triangle(intersec, p0, p1, p2, f, bc_coord)) return t;
		else return numeric_limits<float>::max();
	}
}

vec3 t_get_texture(const texture& t, const int vt[], float bc_coord[]) {
	vec3 texture_color;
	float u = 0;
	float v = 0;
	for (int i = 0; i < 3; i++) {
		u += bc_coord[i] * VT[vt[i] - 1].x;
		v += bc_coord[i] * VT[vt[i] - 1].y;
	}
	while (u > 1) u -= 1;
	while (v > 1) v -= 1;
	int i = int(round(v * (t.height - 1)));
	int j = int(round(u * (t.width - 1)));
	texture_color = t.t_img[i][j];
	return texture_color;
}

// function to check if a point is in shadow with respect to a light source
bool inshadow(vec3 L, const lightobj& LIGHT, const vector<sphereobj>& S, \
	const vector<face>& F, vec3 intersec, vec3 N) {
	float epsilon = 0.001;
	float min_dis = numeric_limits<float>::max();
	for (int i = 0; i < S.size(); i++) { // for each sphere object
		float dis = intersect_s(L, intersec + epsilon * N, S[i]);
		if (LIGHT.flag == 0 && dis < min_dis) return true;
		else if (LIGHT.flag == 1) {
			float dis2light = (intersec - LIGHT.pod).length();
			if (dis < dis2light) return true;
		}
	}
	float foo[3];
	for (int i = 0; i < F.size(); i++) { // for each triangle object
		float dis = intersect_t(F[i], L, intersec + epsilon * N, foo);
		if (LIGHT.flag == 0 && dis < min_dis) return true;
		else if (LIGHT.flag == 1) {
			float dis2light = (intersec - LIGHT.pod).length();
			if (dis < dis2light) return true;
		}
	}
	return false;
}

// function to compute soft shadow
float shadow(vec3 L, const lightobj& LIGHT, const vector<sphereobj>& S, \
	const vector<face>& F, vec3 intersec, vec3 N) {
	if (LIGHT.flag == 0 || LIGHT.flag == 1) { // no soft shadow for directional light
		if (!inshadow(L, LIGHT, S, F, intersec, N)) return 1;
		else return 0;
	}
	int num_of_ray = 30;
	float f = 0; // proportion of rays that reaches the light source
	default_random_engine generator;
	uniform_real_distribution<float> distribution(-0.05, 0.05);
	for (int i = 0; i < num_of_ray; i++) {
		lightobj l = LIGHT;
		l.pod.x += distribution(generator);
		l.pod.y += distribution(generator);
		l.pod.z += distribution(generator);
		L = l.pod - intersec;
		L.normalize();
		if (!inshadow(L, l, S, F, intersec, N)) f += 1.0;
	}
	f /= num_of_ray;
	return f;
}

float shadow_new(vec3 L, const lightobj& LIGHT, const vector<sphereobj>& S, \
	const vector<face>& F, vec3 intersec, vec3 N) {
	float epsilon = 0.001;
	float min_dis = numeric_limits<float>::max();
	// first check the opaque objects
	for (int i = 0; i < opaqsphere.size(); i++) { // for each sphere object
		float dis = intersect_s(L, intersec + epsilon * N, S[opaqsphere[i]]);
		if (LIGHT.flag == 0 && dis < min_dis) return 0;
		else if (LIGHT.flag == 1) {
			float dis2light = (intersec - LIGHT.pod).length();
			if (dis < dis2light) return 0; // completely in shadow
		}
	}
	float foo[3];
	for (int i = 0; i < opaqface.size(); i++) { // for each triangle object
		float dis = intersect_t(F[opaqface[i]], L, intersec + epsilon * N, foo);
		if (LIGHT.flag == 0 && dis < min_dis) return 0;
		else if (LIGHT.flag == 1) {
			float dis2light = (intersec - LIGHT.pod).length();
			if (dis < dis2light) return 0;
		}
	}
	// and then the transparent ones
	float shadow_ext = 1;
	for (int i = 0; i < transphere.size(); i++) { // for each sphere object
		float dis = intersect_s(L, intersec + epsilon * N, S[transphere[i]]);
		if (LIGHT.flag == 0 && dis < min_dis) shadow_ext *= (1 - S[transphere[i]].mtlclr.alpha);
		else if (LIGHT.flag == 1) {
			float dis2light = (intersec - LIGHT.pod).length();
			if (dis < dis2light) shadow_ext *= (1 - S[transphere[i]].mtlclr.alpha); // completely in shadow
		}
	}
	for (int i = 0; i < transface.size(); i++) { // for each triangle object
		float dis = intersect_t(F[transface[i]], L, intersec + epsilon * N, foo);
		if (LIGHT.flag == 0 && dis < min_dis) shadow_ext *= (1 - M_COLOR[F[transface[i]].mc_index].alpha);
		else if (LIGHT.flag == 1) {
			float dis2light = (intersec - LIGHT.pod).length();
			if (dis < dis2light) shadow_ext *= (1 - M_COLOR[F[transface[i]].mc_index].alpha);
		}
	}
	return shadow_ext;
}

// attenuation function
float attenuation(float d, const lightobj& L) {
	float f_att = 1 / (L.c.x + L.c.y * d + L.c.z * d * d);
	return f_att;
}

// function to check if a point can be illuminated by a spotlight
float spotlightill(const lightobj& L, const vec3& intersec) {
	float f;
	vec3 V1 = L.dir;
	vec3 V2 = intersec - L.pod;
	V1.normalize();
	V2.normalize();
	f = 180 * acos(V1 * V2) / (PI * L.theta);
	if (f < 0.9) return 1.0;
	if (f <= 1) return (-10 * f + 10);
	else return 0;
}

// function to calculate color for each pixel (ray)
vec3 comp_color(bool is_t, int index, vec3 N, \
	vector<vec3> Ls, vec3 V, vec3 intersec, \
	const vector<sphereobj>& S, const vector<face>& F,\
	const vector<lightobj>& LIGHT, float bc_coord[]) {
	// Note that Ls is an array of L vectors
	mtlcolor mtlclr;
	if (!is_t) mtlclr = S[index].mtlclr;
	else mtlclr = M_COLOR[F[index].mc_index]; // obtain material color

	vec3 color; // initialize the color
	float ka = mtlclr.k.x;
	float kd = mtlclr.k.y;
	float ks = mtlclr.k.z;
	float n = mtlclr.n;
	vec3 amb; // ambient term
	if (!is_t) {
		if (S[index].t == -1) amb = mtlclr.d_color; // sphere is not textured
		else amb = s_get_texture(TT[S[index].t], N, S[index]);
	}
	else {
		if (F[index].texture_pvd) amb = t_get_texture(TT[F[index].t_index], \
			F[index].texture, bc_coord);
		else amb = mtlclr.d_color;
	}
	vec3 diff; // initialize diffuse term
	vec3 spec; // initialize specular term
	for (int i = 0; i < Ls.size(); i++) { // for every light source
		vec3 L = Ls[i];
		float slf = 1; // spotlight factor
		if (LIGHT[i].theta != 0) {
			slf = spotlightill(LIGHT[i], intersec);
			if (slf == 0) continue;
		} // spotlight check
		//float ss = shadow(L, LIGHT[i], S, F, intersec, N); // compute soft shadow
		float ss = shadow_new(L, LIGHT[i], S, F, intersec, N);
		if (ss == 0) continue; // completely in shadow
		float NL = N * L;
		if (NL < 0) NL = 0;
		vec3 H = L - V;
		H.normalize();
		float NH = N * H;
		if (NH < 0) NH = 0;

		vec3 dclr = amb;
		vec3 sclr = mtlclr.s_color;
		dclr.x *= LIGHT[i].color.x;
		dclr.y *= LIGHT[i].color.y;
		dclr.z *= LIGHT[i].color.z;
		sclr.x *= LIGHT[i].color.x;
		sclr.y *= LIGHT[i].color.y;
		sclr.z *= LIGHT[i].color.z;
	
		float f_att = 1;
		if (LIGHT[i].flag == 1) { // point/spot light has attenuation
			float d = (LIGHT[i].pod - intersec).length(); // distance from light to intersection
			f_att = attenuation(d, LIGHT[i]); // attenuation factor
		}

		diff = diff + slf * NL * ss * f_att * dclr;
		spec = spec + slf * pow(NH, n) * ss * f_att * sclr;
	}
	color = ka * amb + kd * diff + ks * spec;
	return color;
}

// function to trace rays
vec3 traceray(const vector<sphereobj>& S, const vector<face>& F, const vec3& ray, \
	 const vec3& origin, const vector<lightobj>& L, raytype rt, int rec_depth) {
	
	vec3 color;
	color.x = -1; // as an indication of a ray not hitting anything
	rec_depth += 1;
	if (rec_depth > 5) return color;

	float closest_d = numeric_limits<float>::max();
	int index = 0;
	for (int sph = 0; sph < S.size(); sph++) { // for each sphere object
		float d;
		d = intersect_s(ray, origin, S[sph]);
		if (d > 0 && d < closest_d) {
			closest_d = d;
			index = sph;
		}
	}
	bool is_t = false; // nearest hit is triangle
	float tmp[3];
	float bc_coord[3]; // barycentric coordinates
	for (int fac = 0; fac < F.size(); fac++) { // for each triangle
		float d;
		d = intersect_t(F[fac], ray, origin, tmp);
		if (d < closest_d) {
			closest_d = d;
			index = fac;
			for (int c = 0; c < 3; c++) {
				bc_coord[c] = tmp[c];
			}
			is_t = true;
		}
	}

	if (closest_d < numeric_limits<float>::max()) { // view intersects an object
		vec3 V;
		vec3 intersec;
		V = ray;
		intersec = origin + closest_d * V;
		vec3 N;
		float F0;
		float alpha;
		float eta;
		if (!is_t) {
			N = intersec - S[index].position; // only for spheres
			N.normalize();
			F0 = S[index].mtlclr.F0;
			eta = S[index].mtlclr.eta;
			alpha = S[index].mtlclr.alpha;
			if (rt == inside && eta != 0 && N * V != 0) { // ray currently inside a surface
				eta = 1 / eta; 
				alpha = 0;
			}
			if (N * V > 0 && eta != 0) N = -N;
		}
		else {
			if (!F[index].normal_pvd) N = F[index].normal;
			else {
				N = bc_coord[0] * VN[F[index].vnormal[0] - 1]\
					+ bc_coord[1] * VN[F[index].vnormal[1] - 1]\
					+ bc_coord[2] * VN[F[index].vnormal[2] - 1];
				N.normalize();
			}
			F0 = M_COLOR[F[index].mc_index].F0;
			eta = M_COLOR[F[index].mc_index].eta;
			alpha = M_COLOR[F[index].mc_index].alpha;
			if (rt == inside && eta != 0 && N * V != 0) { // ray currently inside a surface
				eta = 1 / eta;
				alpha = 0;
			}
			if (N * V > 0 && eta != 0) N = -N;
		}
		vector<vec3> Ls; // an array of vector Ls
		for (int i = 0; i < L.size(); i++) {
			if (L[i].flag == 0) Ls.push_back(-L[i].pod); // directional light
			else Ls.push_back(L[i].pod - intersec); // point light
			Ls[i].normalize();
		}

		color = comp_color(is_t, index, N, Ls, V, intersec, S, F, L, bc_coord);

		if (eta != 0) {
			// recursive calls to calculate reflection and refraction
			vec3 I = -V;
			float costhetai = N * I;
			float Fr = F0 + (1 - F0) * (1 - pow(costhetai, 5));
			// reflection
			vec3 R = 2 * (costhetai) * N - I;
			vec3 o = intersec + N * 0.001;
			vec3 reflection_color = traceray(S, F, R, o, L, rt, rec_depth);
			if (reflection_color.x != -1) color = color + Fr * reflection_color;
			// refraction
			// not opaque
			// no total internal reflection
			if (alpha != 1 && !(sqrt(1 - costhetai* costhetai) > eta)) { 
				vec3 T = (-N) * sqrt(1 - (1 / eta) * (1 / eta) * (1 - costhetai * costhetai)) + (1 / eta) * (costhetai * N - I);
				o = intersec - N * 0.001;
				if (rt == inside) rt = outside;
				else rt = inside;
				vec3 refraction_color = traceray(S, F, T, o, L, rt, rec_depth);
				if (refraction_color.x != -1) color = color + (1 - Fr) * (1 - alpha) * refraction_color;
			}
		}
		
		if (color.x > 1) color.x = 1;
		if (color.y > 1) color.y = 1;
		if (color.z > 1) color.z = 1;
	}
	return color;
}


// function to return color for each pixel
void renderscene(const vector<sphereobj>& S, const vector<face>& F, vector<vector<vec3>>& img, \
	const viewing_prm& v, const vector<vector<vec3>>& rays, const vector<lightobj>& L, const vector<vector<vec3>>& background) {
	for (int i = 0; i < v.img_h; i++) {
		for (int j = 0; j < v.img_w; j++) { // for each ray
			vec3 color = traceray(S, F, rays[i][j], v.eye, L, outside, 0);
			if (color.x != -1) img[i][j] = img[i][j] + color;
			else img[i][j] = img[i][j] + background[i][j];
		}
	}
}



// function to write to the ppm
void writeppm(string filename, int w, int h, const vector<vector<vec3>>& img) {
	ofstream myimg;
	myimg.open(filename);
	myimg << "P3" << endl;
	myimg << "# RayTracer" << endl;
	myimg << w << " " << h << endl;
	myimg << 255 << endl;
	for (int i = 0; i < h; i++) { // draw to the ppm file
		for (int j = 0; j < w; j++) {
			myimg << int(255 * img[i][j].x) << " ";
			myimg << int(255 * img[i][j].y) << " ";
			myimg << int(255 * img[i][j].z) << "  ";
		}
		myimg << endl;
	}
	myimg.close();
}

int main(int argc, char* argv[]) {
	if (argc != 2) {
		cout << "No input file specified" << endl;
		exit(1);
	}

	// read the decription from the file
	string content = readfile(argv[1]);
	stringstream desc(content);
	string keyword;

	viewing_prm v; // create an object of view
	mtlcolor mtlclr; // create an object of mtlcolor

	vector<sphereobj> S; // create an empty array of sphere objects
	//vector<ellipsoidobj> E; // create an empty array of ellipsoid objects
	vector<lightobj> L; // create an empty array of light objects
	vector<face> F; // array of faces
	vec3 bkgcolor;

	bool f0 = false; // flags to keep track of the initialization of prms
	bool f1 = false;
	bool f2 = false;
	bool f3 = false;
	bool f4 = false;
	bool f5 = false;
	bool f6 = false;
	bool f7 = false;

	// adjust the viewing parameters according to the description
	while (!desc.eof()) {
		desc >> keyword;
		if (keyword == "eye") {
			desc >> v.eye.x >> v.eye.y >> v.eye.z;
			f0 = true;
		}
		else if (keyword == "viewdir") {
			desc >> v.viewdir.x >> v.viewdir.y >> v.viewdir.z;
			f1 = true;
		}
		else if (keyword == "updir") {
			desc >> v.updir.x >> v.updir.y >> v.updir.z;
			f2 = true;
		}
		else if (keyword == "bkgcolor") {
			desc >> bkgcolor.x >> bkgcolor.y >> bkgcolor.z;
			if (bkgcolor.x > 1 || bkgcolor.y > 1 || bkgcolor.z > 1 || \
				bkgcolor.x < 0 || bkgcolor.y < 0 || bkgcolor.z < 0) {
				cout << "Invalid background color" << endl;
				exit(3);
			}
			f3 = true;
		}
		else if (keyword == "mtlcolor") {
			desc >> mtlclr.d_color.x >> mtlclr.d_color.y >> mtlclr.d_color.z;
			desc >> mtlclr.s_color.x >> mtlclr.s_color.y >> mtlclr.s_color.z;
			desc >> mtlclr.k.x >> mtlclr.k.y >> mtlclr.k.z;
			desc >> mtlclr.n >> mtlclr.alpha >> mtlclr.eta;
			if (!mtlclr.isvalid()) {
				cout << "Invalid material color" << endl;
				exit(3);
			}
			M_COLOR.push_back(mtlclr);
			f4 = true;
		}
		else if (keyword == "sphere") {
			sphereobj s;
			desc >> s.position.x >> s.position.y >> s.position.z;
			desc >> s.r;
			s.mtlclr = mtlclr;
			if (TT.size() != 0) s.t = TT.size() - 1;

			if (!(s.isvalid())) {
				cout << "The description of the sphere object is invalid" << endl;
				exit(4);
			}
			f5 = true;
			S.push_back(s);
			if (s.mtlclr.alpha == 1) opaqsphere.push_back(S.size() - 1);
			else transphere.push_back(S.size() - 1);
		}
		else if (keyword == "vfov") {
			desc >> v.vfov;
			f6 = true;
		}
		else if (keyword == "imsize") {
			desc >> v.img_w >> v.img_h;
			f7 = true;
		}
		else if (keyword == "light" || keyword == "attlight" || keyword == "spotlight"\
			|| keyword == "attspotlight") {
			lightobj l;
			desc >> l.pod.x >> l.pod.y >> l.pod.z;
			if (keyword == "spotlight" || keyword == "attspotlight") {
				desc >> l.dir.x >> l.dir.y >> l.dir.z;
				desc >> l.theta;
			}
			else desc >> l.flag;
			desc >> l.color.x >> l.color.y >> l.color.z;
			if (keyword == "attlight" || keyword == "attspotlight") {
				desc >> l.c.x >> l.c.y >> l.c.z;
			}
			if (l.isvalid()) L.push_back(l);
			else {
				cout << "The description of the light object is invalid" << endl;
				exit(4);
			}
		}
		else if (keyword == "v") {
			vec3 v;
			desc >> v.x >> v.y >> v.z;
			VE.push_back(v);
		}
		else if (keyword == "f") {
			face f;
			string s[3];
			desc >> s[0] >> s[1] >> s[2];
			readface(f, s);
			f.mc_index = M_COLOR.size() - 1; // index of mtlcolor
			f.t_index = TT.size() - 1;
			F.push_back(f);
			if (M_COLOR[f.mc_index].alpha == 1) opaqface.push_back(F.size() - 1);
			else transface.push_back(F.size() - 1);
		}
		else if (keyword == "vn") {
			vec3 vn;
			desc >> vn.x >> vn.y >> vn.z;
			VN.push_back(vn);
		}
		else if (keyword == "vt") {
			vec3 t_coord;
			desc >> t_coord.x >> t_coord.y;
			VT.push_back(t_coord);
		}
		else if (keyword == "texture") {
			string texturefile;
			desc >> texturefile;
			texture t(texturefile);
			TT.push_back(t);
		}
		else if (keyword == "viewdist") {
			desc >> D;
			depthofview = true;
			if (D <= 0) {
				cout << "invalid viewing distance" << endl;
				exit(6);
			}
		}
		else {
			cout << "Undefined Keyword: " << keyword << endl;
			exit(1);
		}
	}

	// check if all the prms have been initialized
	if (!(f0 && f1 && f2 && f3 && f4 && f6 && f7)) {
		cout << "The description is incomplete" << endl;
		exit(5);
	}

	// check the validity of the viewing parameters and scene description
	if (!(v.isvalid())) {
		cout << "The description of the viewing parameter is invalid" << endl;
		exit(6);
	}

	vector<vector<vec3>> \
		rays(v.img_h, vector<vec3>(v.img_w, vec3(0, 0, 0)));
	vector<vector<vec3>> \
		background(v.img_h, vector<vec3>(v.img_w, vec3(0, 0, 0)));
	vector<vector<vec3>> \
		img(v.img_h, vector<vec3>(v.img_w, vec3(0, 0, 0)));

	ETA.push(1.0);

	vec3 ul;
	vec3 vstep;
	vec3 hstep;

	computevw(v, ul, vstep, hstep); // compute viewing window
	computerays(v, ul, vstep, hstep, rays); // compute the equations of the rays
	drawbkg(bkgcolor, background, v); // draw the background
	renderscene(S, F, img, v, rays, L, background); // trace ray and draw the objects
	
	if (depthofview) {
		int raynum = 10;
		default_random_engine generator;
		uniform_real_distribution<float> distribution(-0.05, 0.05);
		for (int k = 0; k < raynum - 1; k++) {
			viewing_prm v_new = v;
			v_new.eye.x += distribution(generator);
			v_new.eye.y += distribution(generator);
			v_new.eye.z += distribution(generator);
			computerays(v_new, ul, vstep, hstep, rays); // compute the equations of the rays
			renderscene(S, F, img, v_new, rays, L, background); // trace ray and draw the objects
		}
		for (int i = 0; i < v.img_h; i++) {
			for (int j = 0; j < v.img_w; j++) { // for each ray
				img[i][j].x /= raynum;
				img[i][j].y /= raynum;
				img[i][j].z /= raynum;
			}
		}
	}

	writeppm(string(argv[1]).substr(0, string(argv[1]).rfind(".")) + ".ppm", v.img_w, v.img_h, img); // write to ppm file

	return 0;
}
