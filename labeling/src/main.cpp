#include <igl/readOBJ.h>
#include <igl/png/readPNG.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/decimate.h>
#include <imgui/imgui.h>
#include <sstream>
#include <vector>
#include <fstream>
#include <boost/filesystem.hpp>
#include "landmarks.h"

using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;
namespace fs = boost::filesystem;

bool load_mesh(int);

Viewer viewer;

int current_mesh;
std::vector<std::string> faces_to_label;
std::vector<int> labels; // Indices to verties in low resolution mesh.

// Viewer related point data.
Eigen::MatrixXd points(0, 3);
std::vector<std::string> point_labels;

// Vertex array, #V x 3
Eigen::MatrixXd V(0, 3);
Eigen::MatrixXd V_hd(0, 3);
// Face array, #F x 3
Eigen::MatrixXi F(0, 3);
Eigen::MatrixXi F_hd(0, 3);
// This will contain indices which map vertices of the low resolution mesh to high resolution.
Eigen::VectorXi v_map(0);
Eigen::VectorXi f_map(0);

Eigen::MatrixXd N(0, 3);
Eigen::MatrixXd TC(0, 2);
Eigen::MatrixXd FN(0, 1);
Eigen::MatrixXi FTC(0, 3);

using ColorMap = Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>;
ColorMap R, G, B, A;

fs::path landmark_path(std::string mesh_path) {
  fs::path current_obj(mesh_path);
  fs::path parent_path = current_obj.parent_path();
  fs::path filename = current_obj.stem();
  filename += ".landmarks";
  return parent_path / filename;
}

void save_points() {
  fs::path label_path = landmark_path(faces_to_label[current_mesh]);
  assert(labels.size() == points.rows());
  landmarks::save_points(label_path, labels, V_hd);
}

void points_changed() {
  viewer.data().clear_labels();
  point_labels.clear();
  for (int i=0; i < points.rows(); i++) {
    std::stringstream ss;
    ss << i;
    point_labels.push_back(ss.str());
  }
  viewer.data().set_labels(points, point_labels);
  viewer.data().labels_strings = point_labels;
  viewer.data().labels_positions = points;
}

void load_landmarks(fs::path& filepath) {
  std::string file = filepath.string();
  auto pair = landmarks::read_landmarks(file);
  std::vector<Eigen::RowVector3d> vertices = pair.first;
  std::vector<int> read_labels = pair.second;
  points.resize(read_labels.size(), 3);
  labels.reserve(read_labels.size());
  for (int i=0; i < read_labels.size(); i++) {
    points.row(i) = vertices[i];
    labels.push_back(read_labels[i]);
  }
  points_changed();
}

void load_texture(std::string& mesh_path) {
  fs::path path(mesh_path);
  fs::path texture_path = path.parent_path() / "texture_map.png";
  igl::png::readPNG(texture_path.string(), R, G, B, A);
}

bool load_mesh(int mesh) {
  current_mesh = mesh;
  igl::readOBJ(faces_to_label[current_mesh], V_hd, TC, N, F_hd, FTC, FN);
  load_texture(faces_to_label[current_mesh]);

  igl::decimate(V_hd, F_hd, 10000, V, F, f_map, v_map);

  viewer.data().clear();
  viewer.data().set_mesh(V_hd, F_hd);
  viewer.data().set_normals(N);
  viewer.data().set_texture(R, G, B, A);
  viewer.data().set_uv(TC, FTC);
  viewer.data().set_normals(FN);
  viewer.core().align_camera_center(V);
  viewer.data().show_lines = false;
  viewer.data().show_texture = true;
  viewer.data().set_colors(Eigen::RowVector3d(1.0, 1.0, 1.0));
  viewer.core().lighting_factor = 0.0;

  points.resize(0, 3);
  labels.clear();

  std::string mesh_path = faces_to_label[current_mesh];
  fs::path point_path = landmark_path(mesh_path);

  if (fs::exists(point_path)) {
    load_landmarks(point_path);
    points_changed();
  }

  return true;
}

std::vector<std::string> get_paths(std::string& data_dir) {
  std::vector<std::string> files;
  for (auto & entry : fs::directory_iterator(data_dir)) {
    fs::path obj_path = entry.path() / "textured.obj";
    files.push_back(obj_path.string());
  }
  return files;
}

void remove_point() {
  labels.pop_back();
  Eigen::MatrixXd new_points(points.rows() - 1, 3);
  new_points.block(0, 0, new_points.rows(), 3) = points.block(0, 0, new_points.rows(), 3);
  points = new_points;
  points_changed();
  save_points();
}

void add_point(int vertex_index) {
  int index_on_high_res = v_map[vertex_index];
  labels.push_back(index_on_high_res);
  Eigen::MatrixXd new_points(points.rows() + 1, 3);
  new_points.block(0, 0, points.rows(), 3) = points;
  new_points.row(points.rows()) = V_hd.row(index_on_high_res);
  points = new_points;
  points_changed();
  save_points();
}

void prev_face() {
  current_mesh--;
  if (current_mesh == -1) current_mesh = faces_to_label.size() - 1;
  load_mesh(current_mesh);
}

void next_face() {
  current_mesh++;
  if (current_mesh == faces_to_label.size()) {
    std::exit(0);
  }
  load_mesh(current_mesh);
}

bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers) {
  if (modifiers & IGL_MOD_CONTROL && key == 'Z') {
    if (labels.size() == 0) return false;
    std::cout << "Undo" << std::endl;
    remove_point();
    return true;
  }
  const char key_tab = '\002';
  if (key == 'X') {
    viewer.data().clear();
    viewer.data().set_mesh(V_hd, F_hd);
  } else if (key == 'C') {
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
  }

  if (modifiers & IGL_MOD_SHIFT && key == key_tab) {
    prev_face();
  } else if (key == key_tab) {
    std::cout << "Moving to next face." << std::endl;
    next_face();
  }
  return false;
}

bool callback_mouse_down(Viewer& viewer, int button, int modifier) {
  if (button == (int) Viewer::MouseButton::Right) return false;
  int face_id, vertex_index;
  Vector3f barrycentric_coordinates;

  int down_mouse_x = viewer.current_mouse_x;
  int down_mouse_y = viewer.core().viewport(3) - viewer.current_mouse_y;
  if (igl::unproject_onto_mesh(Eigen::Vector2f(down_mouse_x, down_mouse_y), viewer.core().view,
                              viewer.core().proj, viewer.core().viewport, V, F, face_id, barrycentric_coordinates)) {
    barrycentric_coordinates.maxCoeff(&vertex_index);
    vertex_index = F(face_id, vertex_index);
    add_point(vertex_index);
    return true;
  }
  return false;
}

Eigen::RowVector3d point_colors(247. / 255., 72. / 255., 37. / 255.);

bool callback_pre_draw(Viewer& viewer) {
  viewer.data().clear_points();
  viewer.data().set_points(points, point_colors);
  return false;
}

void init_viewer(Viewer& viewer) {
  viewer.callback_key_down = callback_key_down;
  viewer.callback_mouse_down = callback_mouse_down;
  //viewer.callback_mouse_move = callback_mouse_move;
  //viewer.callback_mouse_up = callback_mouse_up;
  viewer.callback_pre_draw = callback_pre_draw;
  viewer.data().point_size = 25;
  viewer.core().set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
  viewer.data().show_lines = false;
  viewer.data().label_color = RowVector4f(247., 72., 37., 255.) / 255.0;
  viewer.data().show_labels = true;

  igl::opengl::glfw::imgui::ImGuiMenu menu;
  menu.callback_draw_viewer_window = [&menu](){
    menu.draw_viewer_menu();

  };
  viewer.plugins.push_back(&menu);

  viewer.launch();
}

int main(int argc, char *argv[]) {
  std::string data_dir;
  if (argc != 2) {
    std::cout << "Please provide a data directory. ./labeler <data-dir>" << std::endl;
    exit(1);
  } else {
    data_dir = argv[1];
  }
  auto files = get_paths(data_dir);
  for (std::string &file : files ) {
    if (file.find(".obj") != std::string::npos) {
      faces_to_label.push_back(file);
    }
  }
  load_mesh(0);
  init_viewer(viewer);
}

