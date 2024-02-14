#pragma once

#include "var.hpp"

namespace autodiff {
namespace base {
class eqn {
public:
    eqn(var* v) : v_(v) {
        aliases_ = v_->get_aliases();
    }

    double eval() {
        return v_->eval();
    }

    void print_aliases() {
        std::cout << "created equation" << std::endl;
        for(auto& e: aliases_) {
            std::cout << e.first << std::endl;
            for(auto& v: e.second) {
                std::cout << "\t" << v << std::endl;
            }
        }
    }
private:
    var* v_;
    std::map<const var*,std::vector<std::shared_ptr<var>>> aliases_;
};
}
}
