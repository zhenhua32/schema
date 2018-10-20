from schema_async import Schema, And, Use, Optional
import asyncio

schema = Schema([{
  'name': And(str, len),
  'age': And(Use(int), lambda n: 18 <= n <= 99),
  Optional('gender'): And(str, Use(str.lower), lambda s: s in ('squid', 'kid'))
}])

data = [{
  'name': 'Sue',
  'age': '28',
  'gender': 'Squid'
}, {
  'name': 'Sam',
  'age': '42'
}, {
  'name': 'Sacha',
  'age': '20',
  'gender': 'KID'
}]

loop = asyncio.get_event_loop()

validated = loop.run_until_complete(schema.validate(data))

print(validated)
assert validated == [{
  'name': 'Sue',
  'age': 28,
  'gender': 'squid'
}, {
  'name': 'Sam',
  'age': 42
}, {
  'name': 'Sacha',
  'age': 20,
  'gender': 'kid'
}]
