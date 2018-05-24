def main():
    first = open('test.pred')
    second = open('test_89_40.pred')

    same_lines = 0.0
    for l1, l2 in zip(first, second):
        if l1 == l2:
            same_lines += 1

    print 'same:', same_lines

    second.close()
    first.close()

if __name__ == '__main__':
    main()