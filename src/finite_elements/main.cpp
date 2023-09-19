
#include <iostream>

class T {

    public:

    T(int& a_) : a(a_) {}

    int& a;
};


T func() {

    int a = 10;
    T c = T(a);
    return c;
}


int main(int argc, char** argv) {

    T c = func();
    std::cout << c.a << std::endl;
    return 0;
}
