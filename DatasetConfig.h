#pragma once

#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

inline std::map<std::string, std::string>
ReadDatasetPaths(const std::string &filename) {
    std::ifstream file(filename);

    if (!file)
        throw std::runtime_error(
            "Cannot open dataset config: " + filename);

    std::map<std::string, std::string> paths;
    std::string line;

    auto trim = [](const std::string &s) {
        auto begin = s.find_first_not_of(" \t\r\n");
        if (begin == std::string::npos)
            return std::string{};

        auto end = s.find_last_not_of(" \t\r\n");
        return s.substr(begin, end - begin + 1);
    };

    while (std::getline(file, line)) {
        auto comment = line.find('#');
        if (comment != std::string::npos)
            line.erase(comment);

        auto eq = line.find('=');
        if (eq == std::string::npos)
            continue;

        auto key = trim(line.substr(0, eq));
        auto value = trim(line.substr(eq + 1));

        if (!key.empty() && !value.empty())
            paths[key] = value;
    }

    for (const auto &name : {"ttHH", "HH", "ttbar"}) {
        if (!paths.count(name))
            throw std::runtime_error(
                std::string("Missing dataset path: ") + name);
    }

    return paths;
}
