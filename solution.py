# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def trap(height: list) -> int:
    print(height)
    v = 0
    if len(height) < 2:
        return 0, 0
    last = height[0]
    last_ind = 0
    tmp = 0
    first = True
    for ind_1, n in enumerate(height[1:]):
        ind = ind_1 + 1
        print(ind, n, last, v, tmp)
        if first and last == 0:
            last = n
            last_ind = ind
            if last >0:
                first = False
        elif n >= last:
            v += ( (ind-last_ind-1)*last - tmp )
            last = n
            last_ind = ind
            tmp = 0
        else:
            tmp += n
    return v, last_ind


def listToTree(vals):
    length = len(vals)
    if length == 0:
        return None
    root = TreeNode(vals[0])
    nodelist = [root]
    i = 1
    while True:
        node = nodelist.pop(0)
        if i < length:
            node.left = TreeNode(vals[i]) \
                if vals[i] is not None else None
            nodelist.append(node.left)
            i += 1
        else:
            break

        if i < length:
            node.right = TreeNode(vals[i]) \
                if vals[i] is not None else None
            nodelist.append(node.right)
            i += 1
        else:
            break

    return root


def showTree(root, order="level"):
    vals = []
    nodelist = [root]
    while not nodelist:

        for _ in range(len(nodelist)):
            if order == "level":
                node = nodelist.pop(0)
                vals.append(node.val)
                if node.left is not None:
                    nodelist.append(node.left)
                if node.right is not None:
                    nodelist.append(node.right)
    return vals

def showTreePreorder(root):
    if root is None:
        return []
    vals = []
    if root.left is not None:
        vals.extend(showTreePreorder(root.left))
    vals.append(root.val)
    if root.right is not None:
        vals.extend(showTreePreorder(root.right))
    return vals

def showTreeMidorder(root):
    if root is None:
        return []
    vals = []
    vals.append(root.val)
    if root.left is not None:
        vals.extend(showTreeMidorder(root.left))
    if root.right is not None:
        vals.extend(showTreeMidorder(root.right))
    return vals

def showTreeSuforder(root):
    if root is None:
        return []
    vals = []
    if root.right is not None:
        vals.extend(showTreeSuforder(root.right))
    vals.append(root.val)
    if root.left is not None:
        vals.extend(showTreeSuforder(root.left))
    return vals

def getfromPreMid(vals_pre, vals_mid):
    val_id = {}
    for i, v in enumerate(vals_mid):
        val_id[v] = i

    root_val = vals_pre[0]
    root = TreeNode(root_val)
    ind = val_id[root_val]
    print(root_val, ind)
    left_num = ind
    if left_num > 0:
        root.left = getfromPreMid(
            vals_pre[1:left_num+1], vals_mid[:left_num])
    right_num = len(vals_mid) - ind - 1
    if right_num > 0:
        root.left = getfromPreMid(
            vals_pre[left_num+1:], vals_mid[left_num+1:])
    return root


class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        if root is None:
            return True
        else:
            pass


    def trap(self, height: list) -> int:
        v1, last_ind = trap(height)
        print(v1, last_ind)
        v2, last_ind = trap(height[last_ind:][::-1])
        print(v2, last_ind)
        return v1 + v2

    def levelOrder(self, root: TreeNode) -> list:
        if root is None:
            return []
        nodelist = [root]
        res = []
        while nodelist:
            level = []
            for _ in range(len(nodelist)):
                node = nodelist.pop(0)
                level.append(node.val)
                if node.left is not None:
                    nodelist.append(node.left)
                if node.right is not None:
                    nodelist.append(node.right)
            res.append(level)
        return res

    def strToInt1(self, s) -> int:
        maxint = 2**31 -1
        minint = -2**31
        chars = list(s.strip())
        nums = list("1234567890")
        if len(chars)<1 or \
           (chars[0] != '-' and chars[0] != '+' and chars[0] not in nums):
            return 0
        flag = 1
        if chars[0] == '-' or chars[0] == '+':
            flag = -1 if chars[0] == '-' else 1
            chars.pop(0)
        num = 0
        n = len(chars)
        for i in range(n):
            c = chars.pop(0)
            print(c)
            if c not in nums:
                break
            else:
                if flag>0:
                    if num > maxint//10:
                        return maxint
                    elif num == maxint//10:
                        if int(c) > 7:
                            return maxint
                        else:
                            num = num * 10 + int(c)
                    else:
                        num = num * 10 + int(c)
                elif flag < 0:
                    if -num < minint//10:
                        return minint
                    elif -num == minint//10 + 1:
                        if int(c) > 8:
                            return minint
                        else:
                            num = num * 10 + int(c)
                    else:
                        num = num * 10 + int(c)
        return num * flag

    def strToInt(self, s) -> int:
        maxint = 2**31 -1
        minint = -2**31
        nums = list("1234567890")
        s = s.strip()
        if len(s)<1 or (s[0] != '-' and chars[0] != '+' and s[0] not in nums):
            return 0

        flag = 1
        if s[0] == '-' or s[0] == '+':
            flag = -1 if chars[0] == '-' else 1
        num = 0
        n = len(chars)
        for i in range(n):
            c = chars[i]
            print(i, c)
            if c not in nums:
                break
        chars = chars[:i] if i != n-1 else chars
        if flag > 0:
            if len(chars) > 8:
                if int(''.join(chars[:-1])) > maxint//10 or \
                   (int(''.join(chars[:-1])) == maxint//10 and int(chars[-1])>7):
                    return maxint
                else:
                    return int(''.join(chars))
            else:
                return int(''.join(chars))
        else:
            if len(chars) > 8:
                print(-int(''.join(chars[:-1])) , 1 + minint//10)
                if -int(''.join(chars[:-1])) < minint//10 or \
                   (-int(''.join(chars[:-1])) == 1 + minint//10 and int(chars[-1])>8):
                    return minint
                else:
                    return -int(''.join(chars))
            else:
                return -int(''.join(chars))

    def threeSumClosest(self, nums: list, target: int) -> int:
        nums = sorted(nums)
        length = len(nums)
        minPos = 1000000
        closest = 100000
        for i in range(0, length-2):
            if nums[i] + nums[i+1] - target < minPos:
                for j in range(i+1, length-1):
                    if nums[i] + nums[j] + nums[j+1] - target < minPos:
                        for k in range(j+1, length):
                            s = nums[i] + nums[j] + nums[k] - target
                            if s < minPos:
                                if abs(s) < abs(closest):
                                    closest = s
                            else:
                                break
        return closest + target

    def threeSum(self, nums) -> list:
        length = len(nums)
        if length < 3:
            return []
        nums = sorted(nums)
        ids = []
        rlimit = length - 1
        for i in range(length - 2):
            found = False
            if nums[i] == nums[i-1] and i > 0:
                continue
            if nums[i] > 0:
                break
            r = rlimit
            l = i + 1
            if nums[r] * nums[i] > 0:
                continue
            print(i, l, r)
            while l < r:
                s = nums[r] + nums[l] + nums[i]
                if s == 0:
                    ids.append([nums[i], nums[l], nums[r]])
                    if not found:
                        rlimit = r
                        found = True
                    r -= 1
                    l += 1
                elif s > 0:
                    r -= 1
                else:
                    l += 1
        return ids

    def maxArea(self, heights) -> int:
        length = len(heights)
        if length < 2:
            return 0

        l, r = 0, length-1
        maxArea = 0
        while l < r:
            if heights[l] > heights[r]:
                h = heights[r]
                r -= 1
            else:
                h = heights[l]
                l += 1
            area = (r - l + 1) * h
            maxArea = max(maxArea, area)
        return maxArea

    def zcode(self, s, nrow):
        length = len(s)
        if length <= nrow:
            return s
        ncycle = nrow + nrow - 2
        res = length % ncycle
        times = length // ncycle
        print(times, res)

        rows = []
        rows.append(''.join([s[i*ncycle] for i in range(times)]))
        for irow in range(1, nrow-1):
            rows.append(''.join([s[i*ncycle+irow] for i in range(times)]))
        rows.append(''.join([s[i*ncycle] for i in range(times)]))

        if res > 0:
            for irow in range(res):
                rows[irow].append(s[times*ncycle + irow])
            for i in range(res - nrow):
                rows[nrow - i - 2].append(s[times*ncycle + nrow + i])
        print(rows)







if __name__ == "__main__":
    # height = [0,1,0,2,1,0,1,3,2,1,2,1]
    # height = [4,2,0,3,2,5]
    sol = Solution()
    # v = sol.trap(height)
    # print(v)

    # vals = [3,9,20,None,None,15,7]
    # root = listToTree(vals)
    # sol = Solution()
    # nums = sol.levelOrder(root)
    # root = getfromPreMid(
    #     [9,3,15,20,7],
    #     [9,15,7,20,3]
    # )
    # nums = showTree(root)
    # print(nums)
    # s = "-91283472332"
    # s = "-2147483649"
    # print(sol.strToInt(s))

    nums = [0, 0, 2, 1, -2, -1, 3, -3, 0]
    nums = [0, -1, 1, -1]
    target = 2
    # out = sol.threeSumClosest(nums, target)
    # out = sol.threeSum(nums)
    # out = sol.maxArea([4,3,2,1,4])
    out = sol.zcode("leetcode", 3)


