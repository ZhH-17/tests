from typing import List

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        n1 = len(text1)
        n2 = len(text2)
        if n1 == 0 or n2 == 0:
            return 0
        dp = [[0 for _ in range(n2+1)] for _ in range(n1+1)]
        p2 = 0
        for i in range(1, n1+1):
            p1 = 0
            for j in range(1, n2+1):
                if text2[j-1] not in text1[p1:i]:
                    dp[i][j] = dp[p1][j]
                else:
                    dp[i][j] = dp[p1][j] + 1
                    p1 = text1[p1:i].index(text2[j-1]) + p1 + 1
                print(text1[:i], text2[:j], dp[i][j], p1)
        return dp[-1][-1]


    def findSecondMinimumValue(self, root: TreeNode) -> int:
        res = -1
        choices = [root]
        small = root.val
        while choices:
            n = len(choices)
            for i in range(n):
                ch = choices.pop(0)
                if ch.val > small:
                    res = min(ch.val, res) if res != -1 else ch.val
                if ch.left is not None:
                    choices.append(ch.left)
                    choices.append(ch.right)
        return res

        # def findFirstGt(node, small):
        #     if node.left is None:
        #         return -1
        #     if node.left.val > 

        #     choices = [node]
        #     while choices:
        #         n = len(choices)
        #         for i in range(n):
        #             ch = choices.pop(0)
        #             if ch.val > small:
        #                 return ch.val
        #             else:
        #                 if ch.left is not None:
        #                     choices.append(ch.left)
        #                 if ch.right is not None:
        #                     choices.append(ch.right)
        #     return -1

        # if root.left is None:
        #     return -1
        # v = root.val
        # res = -1
        # val_l = findFirstGt(root.left, v)
        # if val_l != -1:
        #     res = val_l
        # val_r = findFirstGt(root.right, v)
        # if val_r != -1:
        #     res = val_r if res == -1 else min(val_r, res)
        # return res

    def maxFrequency(self, nums: List[int], k: int) -> int:
        if len(nums) == 1:
            return 1
        nums.sort()
        sums = [0 for _ in range(len(nums) + 1)]
        sums[0] = nums[0]
        for i in range(1, len(nums)):
            sums[i] = sums[i-1] + nums[i]
        print(sums)
        l = 0
        r = 1
        tmp_res = 1
        while True:
            print(l, r)
            diff = 0
            diff = sums[r] - sums[l-1]
            need_k = (r - l + 1) * nums[r] - diff
            print(need_k)
            if need_k <= k:
                tmp_res = max(tmp_res, r - l + 1)
                r += 1
            else:
                l += 1
                r = max(l, r+1)
            if r > len(nums) - 1:
                break

        return tmp_res

    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if headA is None or headB is None:
            return None
        if headA == headB:
            return headA
        p1 = headA
        p2 = headB
        change = False
        while True:
            if change and p1.next is None and p2.next is None:
                return None
            p1 = p1.next
            if p1 is None:
                change = True
                p1 = headB

            p2 = p2.next
            if p2 is None:
                change = True
                p2 = headA
            if p1 == p2:
                return p1

    def minOperations(self, target: List[int], arr: List[int]) -> int:
        if len(target) == 0:
            return 0
        length = len(arr)
        indexes = []
        val = target[0]
        if val in arr:
            indexes.append(arr.index(val))
        else:
            indexes.append(0)
            arr.insert(0, val)
        print(arr)
        for i in range(1, len(target)):
            if target[i] in arr[indexes[-1]+1:]:
                indexes.append(indexes[-1]+1 + arr[indexes[-1]+1:].index(target[i]))
            else:
                indexes.append(indexes[-1]+1)
                arr.insert(indexes[-1]+1, target[i])
            print(arr, indexes)
        return len(arr) - length

    def iou(rect, rects):
        ious = []
        x1, y1, x2, y2 = rect
        area_rect = (x2-x1) * (y2-y1)
        for r in rects:
            l = max(x1, r[0])
            u = max(y1, r[1])
            r = min(x2, r[2])
            d = min(y2, r[3])
            area = (r-l) * (d-u)
            if area < 0:
                area = 0
                ious.append(0)
            ious.append(area / (area_rect + (r[2]-r[0])*(r[3]-r[1])) - area)
        return np.array(ious)

    def nms(rects, confs, th):
        rects_new = []
        rects_del = []
        n = len(rects)
        while len(rects)>0:
            ind = np.argmax(confs)
            rect_max = rects.pop(ind)
            confs.pop(ind)

            rects_new.append(rect_max)
            ious = get_iou(rect_max, rects)
            rects = rects[ious<th]
            confs = confs[ious<th]
        return 0

    def traverse(self, root, order='f'):
        # recursive
        if root is None:
            return []
        vals = []
        vals.extend([root.val])
        vals.extend(self.traverse(root.left))
        vals.extend(self.traverse(root.right))

        # dfs
        if root is None:
            return []
        vals = []
        choices = [root]
        while choices:
            ch = choices.pop(-1)
            vals.append(ch.val)
            if ch.right is not None:
                choices.append(ch.right)

            if ch.left is not None:
                choices.append(ch.left)


        return vals

    def constructFromPrePost(self, preorder, postorder):
        n = len(preorder)
        vals = []
        if len(preorder) < 1:
            return []
        root = preorder[0]
        vals.append(root)
        # choices = [root]
        # while choices:
        #     n = len(choices)
        #     for i in range(n):
        #         ch = choices.pop(0)
        #         vals.append(ch)
        #         ind1 = preorder.index(root)
        #         ind2 = postorder.index(root)
        #         if len(vals) == n:
        #         break

        if len(preorder) < 2:
            return vals

        root_left = preorder[1]
        ind = postorder.index(root_left)
        post_left = postorder[:ind+1]
        pre_left = preorder[1:ind+2]
        post_right = postorder[ind+1:-1]
        pre_right = preorder[ind+2:]
        print('left ', pre_left, post_left)
        print('right', pre_right, post_right)
        vals.append(self.constructFromPrePost(pre_left, post_left))
        print('1', vals)
        vals.append(self.constructFromPrePost(pre_right, post_right))
        if len(vals) > 1:
            vals = vals[:1] + vals[1] + vals[2]
        print('2', vals)
        def recover(nums):
            if len(nums) == 1:
                return nums
            else:
                return nums[:1] + recover(nums[1]) + recover(nums[2])
        vals = recover(vals)

        return vals

    def countBits(self, n: int) -> List[int]:
        if n == 0:
            return [0]
        if n == 1:
            return [0, 1]
        res = n
        deg = 0
        while res > 0:
            res = res >> 1
            deg += 1
        ans = [0, 1]
        for i in range(2, deg):
            ans.extend([a+1 for a in ans])
        res = n - 2**(deg-1) + 1
        ans.extend([a+1 for a in ans[:res]])
        return ans

    def movingCount(self, m: int, n: int, k: int) -> int:
        pass

    def maxMatch(m, n, matched):
        p = [-1 for _ in range(n)] # choice of n
        vis = [0 for _ in range(n)] # occupied of n
        def is_match(i):
            pass
        pass


if __name__ == "__main__":
    sol = Solution()
    nums = [1, 2, 4]
    target = [16,7,20,11,15,13,10,14,6,8]
    arr = [11,14,15,7,5,5,6,10,11,6]

    k = 5
    # res = sol.maxFrequency(nums, k)
    # res = sol.minOperations(target, arr)
    # res = sol.longestCommonSubsequence("abs", 'sadh')
    pre = [1,2,4,5,3,6,7]
    post = [4,5,2,6,7,3,1]
    # res = sol.constructFromPrePost(pre, post)
    res = sol.countBits(5)
    print(res)


def fun():
    df = df.applymap(lambda x: x.decode("utf-8", errors="ignore")
                        if isinstance(x, bytes) else x)
    if bd_status == 'A':
        df = df[df[status_name[dev_type]] == 'A,']
    elif bd_status == 'V':
        df = df[df[status_name[dev_type]] != 'A,']

    df = df.sort_values(['collectionDate'], ascending=False)
    data_gps = df.apply(lambda x:max(x) / 2, axis=1)
