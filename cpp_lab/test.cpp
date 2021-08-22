// C++ program to illustrate the
// iterators in vector
#include <iostream>
#include <vector>
  
using namespace std;

vector<int> twoSum(vector<int>& nums, int target) {
    for (auto i = nums.begin(); i != nums.end(); ++i)
        cout << *i << " ";
    // std::cout << target << std::endl;
    return nums;
}
  
int main()
{
    vector<int> nums;
    for(int i = 1; i <= 5; i++) 
        nums.push_back(i);
    // auto r = twoSum( 
    cout << 10 << endl;
  
    return 0;
}
