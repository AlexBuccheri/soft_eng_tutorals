# Add source directories
set(source_directories
        optimisers
        test_functions
    )

foreach (dir IN LISTS source_directories)
    add_subdirectory(${dir})
endforeach ()
