#ifndef LANDMARKS_H
#define LANDMARKS_H
#include <vector>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

namespace landmarks {
  std::pair<std::vector<Eigen::RowVector3d>, std::vector<int>> read_landmarks(std::string filepath) {
    std::ifstream landmark_file;
    landmark_file.open(filepath);
    std::string line;
    int index;
    double v0, v1, v2;
    std::vector<Eigen::RowVector3d> vertices;
    std::vector<int> labels;
    while (std::getline(landmark_file, line)) {
      if (line.empty()) continue;
      std::istringstream ss(line);
      ss >> index;
      ss >> v0;
      ss >> v1;
      ss >> v2;
      Eigen::RowVector3d vertex;
      vertex[0] = v0;
      vertex[1] = v1;
      vertex[2] = v2;
      vertices.push_back(vertex);
      labels.push_back(index);
    }
    landmark_file.close();
    return std::make_pair(vertices, labels);
  }

  void save_points(const fs::path& label_path, const std::vector<int>& labels,
      Eigen::MatrixXd& V) {
    std::ofstream landmark_file;
    landmark_file.open(label_path.string(), std::ios::out & std::ios::trunc);
    for (int i=0; i < labels.size(); i++) {
      int label = labels[i];
      auto coordinates = V.row(labels[i]);
      double v0, v1, v2;
      v0 = coordinates[0]; v1 = coordinates[1]; v2 = coordinates[2];
      landmark_file << labels[i] << " " << v0 << " " << v1 << " " << v2 << "\n";
    }
    landmark_file.close();
  }

  void read_vertices(std::vector<std::pair<int, int>> landmarks, const Eigen::MatrixXd& vertices_in, Eigen::MatrixXd& vertices_out) {
      vertices_out.resize(landmarks.size(), 3);
      for (int i=0; i < landmarks.size(); i++) {
          int vertex_id = landmarks[i].first;
          vertices_out.row(i) = vertices_in.row(vertex_id);
      }
  }
}

#endif
