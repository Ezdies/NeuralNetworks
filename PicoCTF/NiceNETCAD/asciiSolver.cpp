#include <iostream>
#include <cstdlib>

int main()
{
  // Get the value of the LINE_COUNT environmental variable
  const char *lineCountStr = getenv("LINE_COUNT");
  if (lineCountStr == NULL)
  {
    std::cerr << "Error: LINE_COUNT environmental variable not set" << std::endl;
    return 1;
  }

  // Convert the string to an integer
  int lineCount = atoi(lineCountStr);

  int a;

  for (int i = 0; i < lineCount; i++)
  {
    std::cin >> a;
    std::cout << (char) a;
  }

  return 0;
}
