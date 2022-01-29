**Notice.** This repository contains an outdated set of Vulkano examples. All examples have migrated to the primary [Vulkano repository](https://github.com/vulkano-rs/vulkano/tree/master/examples) where we keep them up to date with the recent changes in Vulkano. Contribution to the example set is very welcome! For this purpose, you can open a PR directly to the primary repository.

# Vulkano examples

[![Build Status](https://travis-ci.org/vulkano-rs/vulkano-examples.svg?branch=master)](https://travis-ci.org/vulkano-rs/vulkano-examples)

## Running the examples:

```sh
cargo run --bin <example>
```

## Example:

```sh
cargo run --bin triangle
```

If you want to compare performances with other libraries, you should pass the `--release` flag as
well. Rust is pretty slow in debug mode.

## Projects that use vulkano

*   [vulkan-tutorial-rs](https://github.com/bwasty/vulkan-tutorial-rs) - Ports the examples at [vulkan-tutorial.com](https://vulkan-tutorial.com/) to vulkano.
*   [PF Sandbox](https://github.com/rukai/PF_Sandbox) - A Platform Fighter Sandbox featuring a character editor tightly integrated with gameplay.
*   [Vulkano Text](https://github.com/rukai/vulkano-text) - Basic, easy to use text rendering library for vulkano.

## Contributing

Issues are disabled for this repo, please file issues on the [main vulkano repo](https://github.com/vulkano-rs/vulkano) mentioning that you ran the examples from this repo.

Pull requests should also be made against the examples in the main vulkano repo.
The examples in the main repo are copied here when a breaking release occurs.

However if you know of an external open-source project that uses vulkano, is functional and is useful as a reference.
Feel free to make a PR to this repo adding it to the list of "projects that use vulkano".

## License

Licensed under either of
 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
at your option.
