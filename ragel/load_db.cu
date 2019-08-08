#include <list>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <ctime>
#include "../dogqc/include/csv.h"
#include "../dogqc/include/util.h"
#include "../dogqc/include/mappedmalloc.h"

#define LINE_COUNT          // number of lines in the dataset
#define FILENAME ""         // path to csv file containing one string per line

int main()
{
    std::string single_row;

    // Determine the number of characters and the offsets for every row

    std::clock_t start_import = std::clock();

    size_t character_count = 0;
    int *character_offset;
    character_offset = (int *)malloc_memory_mapped_file(sizeof(int) * (LINE_COUNT + 1), "mmdb/book_line_offset");

    io::CSVReader<1> count_reader(FILENAME);

    for (int i = 0; i < LINE_COUNT + 1; i++) {
        count_reader.read_row(single_row);
        character_offset[i] = character_count;
        character_count += single_row.length() + 1;
    }

    std::cout << "character count: " << character_count << std::endl;

    // Copy the content of the file into memory

    char *book_content;
    book_content = (char *)malloc_memory_mapped_file(sizeof(char) * character_count, "mmdb/book_content");

    io::CSVReader<1> character_reader(FILENAME);

    for (int i = 0; i < LINE_COUNT; i++) {
        character_reader.read_row(single_row);
        strcpy(&book_content[character_offset[i]], single_row.c_str());
    }

    // Put information about line and character count in a file

    int *book_meta;
    book_meta = (int *)malloc_memory_mapped_file(sizeof(int) * 2, "mmdb/book_meta");
    book_meta[0] = LINE_COUNT;
    book_meta[1] = character_count;

    std::clock_t stop_import = std::clock();

    std::cout << "csv read: " << (stop_import - start_import) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
}
