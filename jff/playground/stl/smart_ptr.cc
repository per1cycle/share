#include <iostream>

template<typename T>
class share_ptr
{

public:
    share_ptr(T* ptr): count_(1), ptr_(ptr) {}
    share_ptr(): count_(0), ptr_(nullptr) {}

    ~share_ptr()
    {
       if(ptr_ && count_-- == 1)
       {
            std::cout << "~share ptr, count = " << count_ << std::endl;
            delete ptr_;
       } 
    }

public:
    void print_info()
    {
        std::cout << "ref count: " << count_ << ". value: " << *ptr_ << std::endl;
    }

public:
    T& operator =(const share_ptr& other)
    {
        std::cout << "Operator =() called" << std::endl;
        count_++;
        return *ptr_;
    }

    void increase()
    {
        count_ ++;
    }

    void decrease()
    {
        count_ --;
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
share_ptr<T> make_share(share_ptr<T>& other)
{
    other.increase();
    return other;
}

int main()
{
    share_ptr<int> a(new int(42));
    a.print_info();
    share_ptr<int> b(new int(100));
    b.print_info();

    share_ptr<int> c = a;

    a.print_info();
    c.print_info();

    return 0;
}