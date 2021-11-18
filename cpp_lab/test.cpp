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

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> results;
        int i, j;
        for(i = 0; i < nums.size(); ++i)
            for(j = i+1; j < nums.size(); j++)
                if (nums[i] + nums[j] == target) {
                    results.push_back(i);
                    results.push_back(j);
                }
        return results;
    }
};
  
int main()
{
    vector<int> nums;
    for(int i = 1; i <= 5; i++) 
        nums.push_back(i);
    // auto r = twoSum( 
    cout << 10 << endl;
  
    return 0;
}
