#include <iostream>

class unique_ptr
{

};

template<typename T>
class share_count;

template<typename T>
class share_ptr
{

public:
    share_ptr(T foo): count_(1), foo_(foo) {}
    share_ptr(): count_(0), foo_(nullptr) {}


    ~share_ptr()
    {
       if(ptr_ && count_ == 1)
       {
            delete ptr_;
       } 
    }

public:
    void print_info()
    {
        std::cout << 
    }
private:
    T *ptr_;
    int count_;
};

template<typename T>
class share_count
{

};

template <typename T>
share_ptr<T> make_share(T& foo)
{

}

int main()
{
    int foo = 42;
    share_ptr<int> a;


    return 0;
}