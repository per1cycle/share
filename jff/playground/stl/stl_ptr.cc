#include <iostream>
#include <memory>


int main()
{
    int foo = 42;
    std::shared_ptr<int> a = std::make_shared<int>(foo);
    std::cout << "share count of a: " << a.use_count() << std::endl;
    {
    std::shared_ptr<int> b = a;
    std::cout << "share count of b: " << b.use_count() << std::endl;
    std::cout << "share count of a: " << a.use_count() << std::endl;
    }

    std::cout << "share count of a: " << a.use_count() << std::endl;
    return 0;
}