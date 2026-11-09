import sys
import modal

test = modal.App("hello-world") # apps define modal applications and group one or more functions, i think those use app.function() decorators

@test.function()
def foo(i):
    if not i % 2:
        print("hello, even!")
    else:
        print("world, odd!")
    return i + 1

@test.local_entrypoint()
def main():
    # run local
    print(foo.local(1000))
    # run remote
    print(foo.remote(1000))

    # run parallel
    total = 0
    for ret in foo.map(range(200)):
        total += ret
    print(total)






