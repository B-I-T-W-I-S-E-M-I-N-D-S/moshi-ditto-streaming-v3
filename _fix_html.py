with open('static/index.html', encoding='utf-8') as f:
    lines = f.readlines()

# First </html> is at line index 971 (line number 972). Trim everything after.
# Find the exact index of the first </html>
first_close = next(i for i, l in enumerate(lines) if '</html>' in l)
print(f'First </html> at line {first_close + 1}')

good = lines[:first_close + 1]

with open('static/index.html', 'w', encoding='utf-8') as f:
    f.writelines(good)

with open('static/index.html', encoding='utf-8') as f:
    final = f.read()

print(f'Total lines: {final.count(chr(10))}')
print(f'</html> occurrences: {final.count("</html>")}')
print(f'</body> occurrences: {final.count("</body>")}')
print('Last 80 chars:', repr(final[-80:]))
print('Done.')
