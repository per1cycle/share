/**
 * dfs sample from oi wiki
 * https://oi-wiki.org/search/dfs/
 * https://www.luogu.com.cn/problem/P1706
 */

#include <iostream>

/**
 * global variable for solver
 * stat is used for identify if current number is used in calling stack.
 * pos[i] is when calling at depth i, the number of current position.
 */
int stat[10] = {0};
int pos[10] = {0};

/**
 * solver:
 * @cur current dfs position
 * @total_len define the end of calling stack
 */
void solve(int cur, int total_len)
{
    if (cur == total_len)
    {
        for(int i = 0; i < total_len; i ++)
            std::cout << pos[i] << ' ';
		std::cout << std::endl;
    }

	for(int i = 0; i < total_len; i ++)
	{
		/**
		 * this number is being used, so pass.
		 */
		if (stat[i])
		{
			continue;
		}
		else 
		{
			pos[cur] = i + 1;
			stat[i] = 1;
			solve(cur + 1, total_len);
			stat[i] = 0; // reverse the current stat
		}
	}
}

int main()
{
    int n;
    std::cin >> n;    
    solve(0, n);
	
	return 0;
}
