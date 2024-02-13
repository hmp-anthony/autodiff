#pragma once

#include "var.hpp"

namespace autodiff {
namespace base {
class eqn {
public:
    eqn(var&& v) {
        std::cout << "created equation" << std::endl;
        auto aliases = v.get_aliases();
        for(auto& e: aliases) {
            std::cout << e.first << std::endl;
            for(auto& v: e.second) {
                std::cout << "\t" << v << std::endl;
            }
        }

        v.purge_aliases();
    }
};
}
}
