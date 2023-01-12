# autodiff 

This is definitely a work in progress.

To build :
`make`

To test :
`make test`

```
#include <iostream>                                                                            
                                                                                               
#include "gradient.hpp"                                                                        
#include "var.hpp"                                                                             
                                                                                               
using autodiff::base::var;                                                                     
using autodiff::functions::cos;                                                                
                                                                                               
int main() {                                                                                   
    double a_ = 10;                                                                            
    double b_ = 20;                                                                            
                                                                                               
    double gamma_ = 0.01;                                                                      
                                                                                               
    auto cos_ = autodiff::functions::cos();                                                    
                                                                                               
    for(int i = 0; i < 100; ++i){                                                              
        var a(a_);                                                                             
        var b(b_);                                                                             
        // define loss function.                                                               
        auto c = a * a + b * b + cos_(a * a);                                                  
        // obtain gradient of loff function.                                                   
        auto C = autodiff::base::gradient(c);                                                  
        // take a step                                                                         
        a_ = a_ - gamma_*C[a];                                                                 
        b_ = b_ - gamma_*C[b];                                                                 
        std::cout << a_ << std::endl;                                                          
        std::cout << b_ << std::endl;                                                          
    }                                                                                          
    return 0;                                                                                  
}    

```
