# Part 8: Implementing the Agent Class and Toolchain System

## Introduction

In this tutorial, we'll explore the implementation of the Agent class and its toolchain system in Phi-3-MLX. We'll break down the key components of the class and explain how the toolchain functionality is implemented.

## The Agent Class Structure

Let's start by examining the core structure of the Agent class:

```python
class Agent:
    _default_toolchain = """
        prompt = add_code(prompt, codes)
        responses = generate(prompt, images)
        files, codes = execute(responses, step)
        """

    def __init__(self, toolchain=None, enable_api=True, **kwargs):
        self.enable_api = enable_api
        self.kwargs = kwargs if 'preload' in kwargs else kwargs|{'preload':load(**kwargs)}
        self.set_toolchain(toolchain)
        self.reset()

    def __call__(self, prompt:str, images=None):
        # Implementation details
```

The class is designed with a default toolchain and an initializer that sets up the agent's configuration.

## Toolchain Parsing

The `set_toolchain` method is crucial for understanding how toolchains are processed:

```python
def set_toolchain(self, s):
    def _parse_toolchain(s):
        s = s.strip().rstrip(')')
        out_part, fxn_part = s.split('=')
        fxn_name, args_part = fxn_part.split('(')

        return {
            'fxn': eval(fxn_name.strip()),
            'args': [arg.strip() for arg in args_part.split(',')],
            'out': [out.strip() for out in out_part.split(',')]
        }

    def _parse_return(s):
        if 'return ' not in s:
            return ['responses', 'files']
        return [i.strip() for i in s.split('return ')[1].split(',')]

    s = self._default_toolchain if s is None else s
    self.toolchain = [_parse_toolchain(i) for i in s.split('\n') if '=' in i]
    self.list_outs = _parse_return(s)
```

This method does several important things:

1. It parses each line of the toolchain string into a dictionary.
2. It uses `eval` to convert function names into actual function references.
3. It extracts argument names and output variable names.
4. It determines the final outputs of the toolchain.

## Executing the Toolchain

The `__call__` method is where the toolchain is actually executed:

```python
def __call__(self, prompt:str, images=None):
    prompt = prompt.replace('"', '<|api_input|>') if self.enable_api else prompt
    self.ongoing.update({'prompt':prompt})
    if images is not None:
        self.ongoing.update({'images':images})
    for tool in self.toolchain:
        _returned = tool['fxn'](*[self.ongoing.get(i, None) for i in tool['args']], 
                                **{k:v for k,v in self.kwargs.items() 
                                   if k in inspect.signature(tool['fxn']).parameters.keys()})
        if isinstance(_returned, dict):
            self.ongoing.update({k:_returned[k] for k in tool['out']})
        else:
            self.ongoing.update({k:_returned for k in tool['out']})
    self.log_step()
    return {i:self.ongoing.get(i, None) for i in self.list_outs}
```

This method:

1. Prepares the input prompt and images.
2. Iterates through each tool in the toolchain.
3. Executes each function with the appropriate arguments.
4. Updates the ongoing state with the results of each function.
5. Logs the step and returns the final outputs.

## Logging and State Management

The Agent class includes methods for managing its state and logging:

```python
def reset(self):
    self.log = []
    self.ongoing = {'step':0}
    self.user_since = 0

def log_step(self):
    self.log.append({**self.ongoing})
    with open(f'agent_log.json', "w") as f:
        json.dump(self.log, f, indent=4)
    self.ongoing = {k:None if v==[None] else v for k,v in self.ongoing.items()}
    self.ongoing['step']+=1

def end(self):
    self.ongoing.update({'END':'END'})
    self.log_step()
    self.reset()
```

These methods handle:

- Resetting the agent's state.
- Logging each step of the toolchain execution.
- Writing logs to a JSON file.
- Properly ending a session and preparing for the next one.

## Key Implementation Insights

1. **Dynamic Function Calling**: The use of `eval` in parsing the toolchain allows for dynamic function calling, making the system highly flexible.
2. **State Management**: The `ongoing` dictionary acts as a state manager, passing data between different steps of the toolchain.
3. **Argument Matching**: The system dynamically matches function arguments with available data, allowing for flexible function definitions in the toolchain.
4. **Error Handling**: While not explicitly shown, proper error handling should be implemented, especially around the `eval` function and dynamic function calling.
5. **Extensibility**: The system is designed to be easily extended with new functions, as long as they follow the expected input/output pattern.

## Conclusion

The Agent class implementation we've explored offers a way to chain together different AI operations. This approach can be useful for creating more complex AI workflows. With this system, you can:

1. Create custom toolchains that combine existing functions in different ways.
2. Add new functions to the system to expand its capabilities.
3. Manage and debug complex AI workflows more effectively.

This implementation provides a framework for experimenting with and building upon language models like Phi-3, allowing for more flexible and tailored AI applications.
